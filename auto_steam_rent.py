# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import time
import asyncio
import threading
import secrets
import string
import base64
import hmac
import struct
import sys
import subprocess
import importlib
import tempfile
import urllib.parse
import html as _html
import re
from collections import UserList, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha1
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

for pkg, imp in [("aiohttp", "aiohttp"), ("pytz", "pytz"), ("pysteamauth", "pysteamauth"),
                 ("rsa", "rsa"), ("requests", "requests"), ("yarl", "yarl"),
                 ("playwright", "playwright")]:
    try:
        importlib.import_module(imp)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=120)
        except Exception:
            pass

try:
    from playwright.sync_api import sync_playwright as _sync_pw
    _pw_check_needed = False
    try:
        with _sync_pw() as _p:
            _cb = _p.chromium.executable_path
            if not os.path.exists(_cb):
                _pw_check_needed = True
    except Exception:
        _pw_check_needed = True
    if _pw_check_needed:
        subprocess.check_call(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300
        )
except Exception:
    pass

import aiohttp
import rsa
from pytz import timezone
from pydantic import BaseModel
from pysteamauth.auth import Steam as _BaseSteam
from yarl import URL as YarlURL
from cardinal import Cardinal
from FunPayAPI.common.enums import OrderStatuses, MessageTypes
from FunPayAPI.updater.events import NewOrderEvent, NewMessageEvent, OrderStatusChangedEvent
from tg_bot import CBT as _CBT
from telebot.types import InlineKeyboardMarkup as K, InlineKeyboardButton as B

NAME = "Auto Steam Rent"
VERSION = "1.0.0"
CREDITS = "@kewanmov"
DESCRIPTION = "Плагин для автоматической аренды Steam аккаунтов на площадке FunPay"
UUID = "d12da53a-391f-416c-b49c-d57f697f9208"
SETTINGS_PAGE = True
PAGE_SIZE = 8
MAX_ORDERS_STORED = 500
MAX_PROCESSED_IDS = 1000
ORDERS_MAX_AGE_DAYS = 14

logger = logging.getLogger("FPC.AutoSteamRent")

try:
    MOSCOW_TZ = timezone('Europe/Moscow')
except Exception:
    MOSCOW_TZ = timezone('UTC')

_PERIOD_LABELS = {
    1: "1 час", 2: "2 часа", 3: "3 часа", 6: "6 часов",
    12: "12 часов", 24: "1 день", 48: "2 дня", 72: "3 дня", 168: "7 дней"
}

ALL_PERIODS: List[int] = list(_PERIOD_LABELS)

ICON_STATUS = {"FREE": "🟢", "ACTIVE": "👤", "BUSY": "⏳", "ERROR": "❌"}

CODE_COOLDOWN = 5.0

PASSWORD_CHANGE_TIMEOUT = 180

class SteamEmailVerificationRequired(Exception):
    pass

FUNPAY_LOT_URL = "https://funpay.com/lots/offer?id={lot_id}"
FUNPAY_ORDER_URL = "https://funpay.com/orders/{}/"
FUNPAY_CHAT_URL = "https://funpay.com/chat/?node={}"

_CMD_CODE = frozenset(("!steamguard", "!code", "/code", "код", "code"))
_CMD_TIME = frozenset(("!time", "/time", "время", "time"))
_CMD_EXTEND = frozenset(("!extend", "/extend", "продлить", "extend"))
_CMD_STOCK = frozenset(("!stock", "/stock", "наличие", "stock"))

def _safe_err(e: Exception) -> str:
    text = str(e)
    text = re.sub(r'<[^>]+>', '', text)
    text = _html.escape(text)
    return text[:300]

def _period_label(h: int) -> str:
    return _PERIOD_LABELS.get(h, f"{h}ч")

def _format_periods(hours: List[int]) -> str:
    return ", ".join(_period_label(h) for h in sorted(hours))

def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

_DT_FMT = "%Y-%m-%d %H:%M:%S"

def _parse(s: str) -> datetime:
    try:
        return MOSCOW_TZ.localize(datetime.strptime(s, _DT_FMT))
    except Exception:
        return _now()

def _ntag(tag: str) -> str:
    return tag.strip().lower()

def _extract_lot_id(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    m = re.search(r"[?&]id=(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"/offer/?(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{5,})\b", s)
    if m:
        return m.group(1)
    return None

def _remaining_str(end: str) -> str:
    rem = (_parse(end) - _now()).total_seconds()
    if rem <= 0:
        return "Истекло"
    h, m = divmod(int(rem), 3600)
    return f"{h}ч {m // 60}м"

def _gen_password(length: int = 20) -> str:
    alpha = string.ascii_letters + string.digits
    while True:
        pwd = ''.join(secrets.choice(alpha) for _ in range(length))
        if (any(c.isupper() for c in pwd) and any(c.islower() for c in pwd)
                and any(c.isdigit() for c in pwd)):
            return pwd

def _is_on(v: bool) -> str:
    return "🟢" if v else "🔴"

def _get_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "storage", "plugins",
                        "auto_steam_rent", f"{filename}.json" if "." not in filename else filename)

os.makedirs(os.path.dirname(_get_path("x")), exist_ok=True)

def _load_json(filename: str) -> Any:
    p = _get_path(filename)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[AutoSteamRent] Не удалось прочитать {filename}: {e} — используем пустой объект")
        return {}

_file_lock = threading.Lock()

def _save_json(filename: str, data: Any):
    p = _get_path(filename)
    with _file_lock:
        dir_name = os.path.dirname(p)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            if os.path.exists(p):
                os.replace(tmp_path, p)
            else:
                os.rename(tmp_path, p)
        except Exception as _e:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            except Exception as _e2:
                logger.error(f"[AutoSteamRent] Не удалось сохранить {filename}: {_e2}")

class LotsCache(UserList):

    def __init__(self):
        super().__init__()
        self.updated_at: Optional[float] = None

_lots_cache = LotsCache()

_LOTS_CACHE_TTL = 180.0

def _get_cached_lots(c):
    global _lots_cache
    if not _lots_cache or _lots_cache.updated_at is None or (time.time() - _lots_cache.updated_at) >= _LOTS_CACHE_TTL:
        _lots_cache.data.clear()
        _lots_cache.extend(c.account.get_user(c.account.id).get_lots())
        _lots_cache.updated_at = time.time()
    return _lots_cache

def _invalidate_lots_cache():
    global _lots_cache
    _lots_cache.data.clear()
    _lots_cache.updated_at = None

def _toggle_fp_lots_for_tag(c, tag: str, enable: bool) -> List[str]:
    tag = _ntag(tag)
    with _toggling_lock:
        if tag in _toggling_tags:
            return []
        _toggling_tags.add(tag)
    try:
        lot_ids = [lid for lid in SETTINGS.lots
                   if _ntag((SETTINGS.get_lot(lid) or LotConfig(tag="default", hours=24)).tag) == tag]
        toggled = []
        for lid in lot_ids:
            try:
                lf = c.account.get_lot_fields(int(lid))
                if lf.active != enable:
                    lf.active = enable
                    c.account.save_lot(lf)
                    toggled.append(lid)
                    logger.debug(f"[AutoSteamRent] Лот #{lid} {'включён' if enable else 'выключен'} на FunPay")
            except Exception as e:
                logger.warning(f"[AutoSteamRent] Ошибка переключения лота #{lid}: {e}")
        _invalidate_lots_cache()
        return toggled
    finally:
        with _toggling_lock:
            _toggling_tags.discard(tag)

class RentStatus:
    FREE = "FREE"
    BUSY = "BUSY"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"
    FINISHED = "FINISHED"
    REFUND = "REFUND"

class SteamGuard:
    _time_offset: int = 0
    _last_sync: float = 0
    SYNC_INTERVAL: int = 300
    SYMBOLS = "23456789BCDFGHJKMNPQRTVWXY"

    @classmethod
    def sync_time_sync(cls) -> int:
        try:
            import requests as req
            resp = req.post("https://api.steampowered.com/ITwoFactorService/QueryTime/v0001", timeout=10)
            if resp.status_code == 200:
                st = int(resp.json()["response"]["server_time"])
                cls._time_offset = st - int(time.time())
                cls._last_sync = time.time()
        except Exception:
            pass
        return cls._time_offset

    @classmethod
    async def sync_time_async(cls) -> int:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post("https://api.steampowered.com/ITwoFactorService/QueryTime/v0001",
                                  timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        d = await resp.json()
                        cls._time_offset = int(d["response"]["server_time"]) - int(time.time())
                        cls._last_sync = time.time()
        except Exception:
            pass
        return cls._time_offset

    @classmethod
    def _steam_time(cls) -> int:
        return int(time.time()) + cls._time_offset

    @classmethod
    def _seconds_until_next_window(cls) -> int:
        return 30 - (cls._steam_time() % 30)

    @classmethod
    def _generate(cls, shared_secret: str) -> str:
        ts = cls._steam_time()
        tw = ts // 30
        s = shared_secret
        if len(s) % 4:
            s += '=' * (4 - len(s) % 4)
        sb = base64.b64decode(s)
        hr = hmac.new(sb, struct.pack(">Q", tw), sha1).digest()
        o = hr[19] & 0x0F
        v = struct.unpack(">I", hr[o:o + 4])[0] & 0x7FFFFFFF
        c = ""
        for _ in range(5):
            c += cls.SYMBOLS[v % len(cls.SYMBOLS)]
            v //= len(cls.SYMBOLS)
        return c

    @classmethod
    def code_sync(cls, shared_secret: str) -> str:
        if not shared_secret:
            return "NO_SECRET"
        if time.time() - cls._last_sync > cls.SYNC_INTERVAL:
            cls.sync_time_sync()
        try:
            return cls._generate(shared_secret)
        except Exception:
            return "ERROR"

    @classmethod
    async def code_async(cls, shared_secret: str) -> str:
        if not shared_secret:
            return "NO_SECRET"
        if time.time() - cls._last_sync > cls.SYNC_INTERVAL:
            await cls.sync_time_async()
        try:
            return cls._generate(shared_secret)
        except Exception:
            return "ERROR"

def _generate_confirmation_key(identity_secret: str, timestamp: int, tag: str) -> str:
    s = identity_secret
    if len(s) % 4:
        s += '=' * (4 - len(s) % 4)
    sb = base64.b64decode(s)
    data = struct.pack(">Q", timestamp) + tag.encode("utf-8")
    return base64.b64encode(hmac.new(sb, data, sha1).digest()).decode("utf-8")

class CustomSteam(_BaseSteam):

    def __init__(self, login, password, shared_secret, identity_secret, device_id, steamid):
        super().__init__(login=login, password=password, steamid=steamid,
                         shared_secret=shared_secret, identity_secret=identity_secret,
                         device_id=device_id)
        self._login = login
        self._pwd = password

    @property
    def login(self):
        return self._login

    @property
    def password(self):
        return self._pwd

    async def raw_request(self, method: str, url: str, **kw):
        from urllib3.util import parse_url
        parsed = parse_url(url)
        host = parsed.host or "steamcommunity.com"
        try:
            cookies = await self.cookies(host)
        except Exception:
            cookies = {}
        return await self._requests.request(method=method, url=url, cookies=cookies, **kw)

class PasswordChangeParams:

    def __init__(self, s, account, reset, issueid, lost=0, **kwargs):
        self.s = int(s)
        self.account = int(account)
        self.reset = int(reset)
        self.issueid = int(issueid)
        self.lost = int(lost)

def _validate_mafile(mf: dict) -> List[str]:
    missing = []
    for f in ("shared_secret", "identity_secret", "account_name"):
        if not mf.get(f):
            missing.append(f)
    return missing

def _warn_mafile(mf: dict) -> List[str]:
    warn = []
    if not mf.get("device_id"):
        warn.append("device_id")
    if not (mf.get("Session") or {}).get("SteamID"):
        warn.append("Session.SteamID")
    return warn

_acc_pwd_locks: Dict[int, threading.Lock] = {}

_acc_pwd_locks_mutex = threading.Lock()

_pwd_change_lock = threading.Lock()

def _get_acc_lock(acc_id: int) -> threading.Lock:
    with _acc_pwd_locks_mutex:
        if acc_id not in _acc_pwd_locks:
            _acc_pwd_locks[acc_id] = threading.Lock()
        return _acc_pwd_locks[acc_id]

class SteamPasswordChanger:
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    HELP = "https://help.steampowered.com"

    def __init__(self, mafile: dict, current_password: str):
        self.mafile = mafile
        self.current_password = current_password
        self.login = mafile.get("account_name", "")
        self.shared_secret = mafile.get("shared_secret", "")
        self.identity_secret = mafile.get("identity_secret", "")
        self.device_id = mafile.get("device_id", "")
        self.steamid = int((mafile.get("Session") or {}).get("SteamID", 0))
        self._steam: Optional[CustomSteam] = None
        missing = _validate_mafile(mafile)
        if missing:
            raise ValueError(f"Отсутствует в maFile: {', '.join(missing)}")
        full_check = _warn_mafile(mafile)
        if full_check:
            raise ValueError(f"Отсутствует в maFile (нужно для смены пароля): {', '.join(full_check)}")

    async def change_password(self) -> str:
        new_password = _gen_password(20)
        self._steam = CustomSteam(
            login=self.login, password=self.current_password,
            shared_secret=self.shared_secret, identity_secret=self.identity_secret,
            device_id=self.device_id, steamid=self.steamid)
        await self._login_steam()
        params = await self._get_wizard_params()
        logger.info(f"[AutoSteamRent] Wizard params: s={params.s} issueid={params.issueid}")
        await self._playwright_open_wizard(params)
        confirmed = await self._confirm_recovery(params)
        if not confirmed:
            raise Exception(f"Mobile confirmation не принята для {self.login}")
        logger.info(f"[AutoSteamRent] Мобильное подтверждение: {self.login} — OK")
        await self._poll_recovery(params)
        await self._verify_recovery_code(params)
        await self._get_next_step(params)
        key = await self._get_rsa_key()
        enc_old = self._encrypt(self.current_password, key["publickey_mod"], key["publickey_exp"])
        await self._verify_old_password(params, enc_old, key["timestamp"])
        logger.info(f"[AutoSteamRent] Старый пароль подтверждён: {self.login}")
        await self._check_password_available(new_password)
        key2 = await self._get_rsa_key()
        enc_new = self._encrypt(new_password, key2["publickey_mod"], key2["publickey_exp"])
        await self._do_change_password(params, enc_new, key2["timestamp"])
        logger.info(f"[AutoSteamRent] Пароль изменён: {self.login}")
        return new_password

    async def _login_steam(self):
        for attempt in range(3):
            try:
                await SteamGuard.sync_time_async()
                secs_left = SteamGuard._seconds_until_next_window()
                if secs_left < 10:
                    wait = secs_left + 3
                    logger.info(f"[AutoSteamRent] Смена пароля: {self.login} — ожидание TOTP ({wait}с)")
                    await asyncio.sleep(wait)
                    await SteamGuard.sync_time_async()
                await self._steam.login_to_steam()
                logger.info(f"[AutoSteamRent] Авторизация: {self.login} — OK")
                await asyncio.sleep(2)
                for wu in (f"{self.HELP}/en/", "https://steamcommunity.com/my/"):
                    try:
                        await self._steam.raw_request("GET", wu, headers={"User-Agent": self.UA})
                        logger.debug(f"[AutoSteamRent] Warmup OK: {wu}")
                    except Exception as e:
                        logger.debug(f"[AutoSteamRent] Warmup failed {wu}: {e}")
                return
            except Exception as e:
                err = str(e)
                logger.warning(f"[AutoSteamRent] Авторизация попытка {attempt+1}/3: {err[:120]}")
                if "TwoFactorCodeMismatch" in err:
                    wait = SteamGuard._seconds_until_next_window() + 3
                    await asyncio.sleep(wait)
                    await SteamGuard.sync_time_async()
                elif "RateLimitExceeded" in err:
                    await asyncio.sleep(30 * (attempt + 1))
                elif "InvalidPassword" in err:
                    raise Exception(f"Неверный пароль для {self.login}")
                else:
                    if attempt >= 2:
                        raise
                    await asyncio.sleep(5)
        raise Exception(f"Steam login failed после 3 попыток для {self.login}")

    async def _get_wizard_params(self) -> PasswordChangeParams:
        urls = [
            f"{self.HELP}/wizard/HelpChangePassword?redir=store/account/",
            f"{self.HELP}/en/wizard/HelpChangePassword",
        ]
        for url in urls:
            try:
                resp = await self._steam.raw_request(
                    "GET", url,
                    headers={
                        "Accept": "text/html,*/*",
                        "Referer": "https://store.steampowered.com/",
                        "User-Agent": self.UA,
                    },
                    allow_redirects=True
                )
                final_url = ""
                if hasattr(resp, 'url'):
                    final_url = str(resp.url)
                elif hasattr(resp, 'real_url'):
                    final_url = str(resp.real_url)
                history = getattr(resp, "history", []) or []
                logger.debug(f"[AutoSteamRent] WizardParams {url[:55]} -> {final_url[:100]} history={len(history)}")
                all_urls = [final_url] + [str(getattr(h, "url", "")) for h in history]
                for src in all_urls:
                    if "s=" in src and "issueid=" in src:
                        try:
                            q = dict(YarlURL(src).query)
                            if all(k in q for k in ("s", "account", "reset", "issueid")):
                                logger.debug(f"[AutoSteamRent] Params from URL: {q}")
                                return PasswordChangeParams(**q)
                        except Exception as e:
                            logger.debug(f"[AutoSteamRent] URL parse error: {e}")
                try:
                    if hasattr(resp, 'text') and callable(resp.text):
                        html_body = await resp.text()
                    elif isinstance(resp, bytes):
                        html_body = resp.decode("utf-8", errors="replace")
                    elif isinstance(resp, str):
                        html_body = resp
                    else:
                        html_body = ""
                except Exception as e:
                    logger.debug(f"[AutoSteamRent] Read body error: {e}")
                    html_body = ""
                found = {}
                patterns = {
                    "s": [r'[?&]s=(\d+)', r'"s"\s*:\s*(\d+)'],
                    "account": [r'[?&]account=(\d+)', r'"account"\s*:\s*(\d+)'],
                    "reset": [r'[?&]reset=(\d+)', r'"reset"\s*:\s*(\d+)'],
                    "issueid": [r'[?&]issueid=(\d+)', r'"issueid"\s*:\s*(\d+)'],
                }
                for key_name, pats in patterns.items():
                    for pat in pats:
                        m = re.search(pat, html_body)
                        if m:
                            found[key_name] = m.group(1)
                            break
                if all(k in found for k in ("s", "account", "reset", "issueid")):
                    logger.debug(f"[AutoSteamRent] Params from HTML: {found}")
                    return PasswordChangeParams(**found)
            except Exception as e:
                logger.warning(f"[AutoSteamRent] WizardParams URL failed {url}: {e}")
        raise Exception(f"Не удалось получить wizard params для {self.login}")

    async def _playwright_open_wizard(self, params: PasswordChangeParams):
        wizard_url = (
            f"{self.HELP}/en/wizard/HelpWithLoginInfoEnterCode"
            f"?s={params.s}&account={params.account}&reset={params.reset}"
            f"&lost={params.lost}&issueid={params.issueid}"
        )
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("[AutoSteamRent] Playwright не установлен — продолжаем без браузера")
            return
        cookies_for_pw = []
        for domain in ["help.steampowered.com", "store.steampowered.com", "steamcommunity.com"]:
            try:
                dc = await self._steam.cookies(domain)
                if isinstance(dc, dict):
                    for name, value in dc.items():
                        cookies_for_pw.append({
                            "name": name,
                            "value": str(value),
                            "domain": f".{domain}",
                            "path": "/"
                        })
            except Exception as e:
                logger.debug(f"[AutoSteamRent] cookies {domain}: {e}")
        pw = None
        browser = None
        try:
            pw = await async_playwright().start()
            try:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--single-process",
                        "--no-zygote",
                        "--disable-extensions",
                        "--disable-software-rasterizer",
                        "--disable-background-networking",
                    ]
                )
            except Exception as e:
                logger.warning(f"[AutoSteamRent] Chromium не запустился: {e} — продолжаем без браузера")
                return
            context = await browser.new_context(
                user_agent=self.UA,
                locale="en-US",
                viewport={"width": 1280, "height": 720}
            )
            if cookies_for_pw:
                await context.add_cookies(cookies_for_pw)
            page = await context.new_page()
            try:
                await page.goto(wizard_url, wait_until="domcontentloaded", timeout=30000)
                logger.info(f"[AutoSteamRent] Playwright: wizard загружен")
            except Exception as e:
                logger.debug(f"[AutoSteamRent] Playwright wizard goto: {e}")
            await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"[AutoSteamRent] Playwright ошибка (non-critical): {e} — продолжаем без браузера")
        finally:
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass
            if pw:
                try:
                    await pw.stop()
                except Exception:
                    pass

    async def _confirm_recovery(self, params: PasswordChangeParams) -> bool:
        cid_str = str(params.s)
        empty_in_a_row = 0
        for attempt in range(20):
            try:
                await SteamGuard.sync_time_async()
                ts = int(time.time()) + SteamGuard._time_offset
                conf_key = _generate_confirmation_key(self.identity_secret, ts, "getlist")
                getlist_url = (
                    "https://steamcommunity.com/mobileconf/getlist"
                    f"?p={urllib.parse.quote(self.device_id)}"
                    f"&a={self.steamid}"
                    f"&k={urllib.parse.quote(conf_key)}"
                    f"&t={ts}&m=android&tag=getlist"
                )
                try:
                    raw = await self._steam.raw_request(
                        "GET", getlist_url,
                        headers={
                            "Accept": "application/json, text/plain, */*",
                            "User-Agent": self.UA,
                            "X-Requested-With": "com.valvesoftware.android.steam.community",
                        }
                    )
                except Exception as e:
                    logger.warning(f"[AutoSteamRent] getlist request error: {e}")
                    await asyncio.sleep(3)
                    continue
                try:
                    data = await self._parse_response(raw, getlist_url)
                except Exception as e:
                    logger.warning(f"[AutoSteamRent] getlist parse: {e}")
                    await asyncio.sleep(3)
                    continue
                if not data.get("success"):
                    logger.warning(
                        f"[AutoSteamRent] getlist not success "
                        f"(login={self.login}, sid={self.steamid}, dev={self.device_id[:12]}..): {data}"
                    )
                    await asyncio.sleep(3)
                    continue
                confs = data.get("conf", [])
                logger.info(
                    f"[AutoSteamRent] getlist attempt {attempt+1}/20: "
                    f"{len(confs)} confirmation(s) for {self.login}"
                )
                if not confs:
                    empty_in_a_row += 1
                    if empty_in_a_row == 3:
                        logger.warning(
                            f"[AutoSteamRent] {self.login}: {empty_in_a_row} пустых getlist подряд. "
                            "Возможные причины: 1) IP бота не доверен Steam — войди в Steam с этого IP и подтверди письмом; "
                            "2) device_id в maFile неверный; 3) confirmation уже была отклонена."
                        )
                        if tg_logs:
                            tg_logs.error(
                                f"⚠️ {self.login}: Steam не выдаёт подтверждение смены пароля.\n"
                                "Проверьте: 1) IP бота (нужен trusted для этого аккаунта), "
                                "2) device_id в maFile, 3) не заблокирован ли аккаунт."
                            )
                    await asyncio.sleep(3)
                    continue
                empty_in_a_row = 0
                for ci in confs:
                    logger.debug(
                        f"[AutoSteamRent]   conf id={ci.get('id')} type={ci.get('type')} "
                        f"type_name={ci.get('type_name')} creator_id={ci.get('creator_id')} "
                        f"summary={ci.get('summary')}"
                    )
                target = next(
                    (ci for ci in confs if str(ci.get("creator_id", "")) == cid_str),
                    None
                )
                if target is None:
                    for ci in confs:
                        type_id = int(ci.get("type", 0))
                        type_name = str(ci.get("type_name", "")).lower()
                        summary = str(ci.get("summary", "")).lower()
                        if type_id == 6 or any(x in type_name for x in ("recovery", "password", "account")) \
                                or any(x in summary for x in ("recovery", "password", "change")):
                            target = ci
                            logger.debug(f"[AutoSteamRent] fallback by type/summary: {type_name!r} {summary!r}")
                            break
                if target is None and len(confs) == 1:
                    target = confs[0]
                    logger.debug(f"[AutoSteamRent] fallback: единственная confirmation (creator_id={confs[0].get('creator_id')}, expected {cid_str})")
                if target is None:
                    logger.debug(
                        f"[AutoSteamRent] attempt {attempt+1}: creator_id {cid_str} не найден среди "
                        f"{[ci.get('creator_id') for ci in confs]}"
                    )
                    await asyncio.sleep(3)
                    continue
                await asyncio.sleep(1)
                ts2 = int(time.time()) + SteamGuard._time_offset
                allow_key = _generate_confirmation_key(self.identity_secret, ts2, "allow")
                ajaxop_url = (
                    "https://steamcommunity.com/mobileconf/ajaxop"
                    f"?p={urllib.parse.quote(self.device_id)}"
                    f"&a={self.steamid}"
                    f"&k={urllib.parse.quote(allow_key)}"
                    f"&t={ts2}&m=android&tag=allow&op=allow"
                    f"&cid={target['id']}&ck={target['nonce']}"
                )
                try:
                    raw = await self._steam.raw_request(
                        "GET", ajaxop_url,
                        headers={
                            "Accept": "application/json, text/plain, */*",
                            "User-Agent": self.UA,
                            "X-Requested-With": "com.valvesoftware.android.steam.community",
                        }
                    )
                    result = await self._parse_response(raw, ajaxop_url)
                except Exception as e:
                    logger.warning(f"[AutoSteamRent] ajaxop error: {e}")
                    await asyncio.sleep(3)
                    continue
                logger.info(f"[AutoSteamRent] ajaxop result for {self.login}: {result}")
                if result.get("success"):
                    return True
                logger.error(f"[AutoSteamRent] Подтверждение отклонено: {result}")
                return False
            except Exception as e:
                logger.warning(f"[AutoSteamRent] Подтверждение попытка {attempt+1}: {e}")
                await asyncio.sleep(3)
        return False

    async def _get_sessionid(self) -> str:
        try:
            cookies = await self._steam.cookies("help.steampowered.com")
            if isinstance(cookies, dict) and "sessionid" in cookies:
                return cookies["sessionid"]
        except Exception:
            pass
        try:
            return await self._steam.sessionid("help.steampowered.com")
        except Exception:
            pass
        raise Exception("Не удалось получить sessionid для help.steampowered.com")

    async def _parse_response(self, resp, url: str) -> dict:
        if isinstance(resp, bytes):
            text = resp.decode("utf-8", errors="replace")
        elif isinstance(resp, str):
            text = resp
        elif hasattr(resp, 'text') and callable(resp.text):
            text = await resp.text()
        else:
            text = str(resp) if resp is not None else ""
        text = text.strip()
        if not text:
            raise Exception(f"Empty response from {url}")
        if text.startswith("<"):
            low = text.lower()
            if any(s in low for s in ("verify by email", "check your email", "email verification",
                                      "подтвердите по почте", "проверьте почту", "ссылку из письма")):
                raise SteamEmailVerificationRequired(
                    f"Steam требует email-подтверждение для recovery (URL: {url}). "
                    "Залогинься в Steam с этого IP и подтверди вход письмом, затем повтори."
                )
            m = re.search(r'<div[^>]*id=["\']error_description["\'][^>]*>([^<]+)<', text)
            err = m.group(1).strip() if m else text[:150]
            raise Exception(f"HTML response from {url}: {err}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise Exception(f"JSONDecodeError from {url}: {e} | raw: {text[:200]}")

    async def _help_post(self, endpoint: str, data: dict) -> dict:
        url = f"{self.HELP}{endpoint}"
        sid = await self._get_sessionid()
        data["sessionid"] = sid
        try:
            resp = await self._steam.raw_request(
                "POST", url,
                data=data,
                headers={
                    "Accept": "*/*",
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Origin": self.HELP,
                    "Referer": f"{self.HELP}/en/",
                    "User-Agent": self.UA,
                    "X-Requested-With": "XMLHttpRequest",
                }
            )
        except Exception as e:
            raise Exception(f"POST {endpoint} failed: {e}")
        return await self._parse_response(resp, url)

    async def _help_get(self, endpoint: str, params: dict) -> dict:
        sid = await self._get_sessionid()
        params["sessionid"] = sid
        qs = urllib.parse.urlencode(params)
        url = f"{self.HELP}{endpoint}?{qs}"
        try:
            resp = await self._steam.raw_request(
                "GET", url,
                headers={
                    "Accept": "*/*",
                    "User-Agent": self.UA,
                    "X-Requested-With": "XMLHttpRequest",
                }
            )
        except Exception as e:
            raise Exception(f"GET {endpoint} failed: {e}")
        return await self._parse_response(resp, url)

    async def _poll_recovery(self, params: PasswordChangeParams):
        for i in range(15):
            r = await self._help_post(
                "/en/wizard/AjaxPollAccountRecoveryConfirmation",
                {
                    "wizard_ajax": "1",
                    "s": str(params.s),
                    "reset": str(params.reset),
                    "lost": str(params.lost),
                    "method": "8",
                    "issueid": str(params.issueid),
                    "gamepad": "0",
                }
            )
            logger.debug(f"[AutoSteamRent] PollRecovery {i+1}: {r}")
            if r.get("success") or r.get("continue"):
                return
            if r.get("errorMsg"):
                raise Exception(f"PollRecovery: {r['errorMsg']}")
            await asyncio.sleep(2)
        raise Exception("Poll confirmation timed out")

    async def _verify_recovery_code(self, params: PasswordChangeParams):
        r = await self._help_get(
            "/en/wizard/AjaxVerifyAccountRecoveryCode",
            {
                "code": "",
                "s": str(params.s),
                "reset": str(params.reset),
                "lost": str(params.lost),
                "method": "8",
                "issueid": str(params.issueid),
                "wizard_ajax": "1",
                "gamepad": "0",
            }
        )
        logger.debug(f"[AutoSteamRent] VerifyCode: {r}")
        if r.get("errorMsg"):
            raise Exception(f"VerifyCode: {r['errorMsg']}")

    async def _get_next_step(self, params: PasswordChangeParams):
        r = await self._help_post(
            "/en/wizard/AjaxAccountRecoveryGetNextStep",
            {
                "wizard_ajax": "1",
                "s": str(params.s),
                "account": str(params.account),
                "reset": str(params.reset),
                "issueid": str(params.issueid),
                "lost": "2",
            }
        )
        logger.debug(f"[AutoSteamRent] GetNextStep: {r}")
        if r.get("errorMsg"):
            raise Exception(f"GetNextStep: {r['errorMsg']}")

    async def _get_rsa_key(self) -> dict:
        r = await self._help_post(
            "/en/login/getrsakey/",
            {"username": self.login}
        )
        logger.debug(f"[AutoSteamRent] RSA: has_mod={bool(r.get('publickey_mod'))}")
        if not r.get("publickey_mod"):
            raise Exception(f"RSA key missing: {r}")
        return r

    async def _verify_old_password(self, params: PasswordChangeParams, enc_pwd: str, ts: str):
        r = await self._help_post(
            "/en/wizard/AjaxAccountRecoveryVerifyPassword/",
            {
                "s": str(params.s),
                "lost": "2",
                "reset": "1",
                "password": enc_pwd,
                "rsatimestamp": ts,
            }
        )
        logger.debug(f"[AutoSteamRent] VerifyOldPwd: {r}")
        if r.get("errorMsg"):
            raise Exception(f"VerifyOldPassword: {r['errorMsg']}")

    async def _check_password_available(self, password: str):
        r = await self._help_post(
            "/en/wizard/AjaxCheckPasswordAvailable/",
            {
                "wizard_ajax": "1",
                "password": password,
            }
        )
        logger.debug(f"[AutoSteamRent] CheckNewPwd: {r}")
        if not r.get("available"):
            raise Exception(f"Password not available: {r}")

    async def _do_change_password(self, params: PasswordChangeParams, enc_pwd: str, ts: str):
        r = await self._help_post(
            "/en/wizard/AjaxAccountRecoveryChangePassword/",
            {
                "wizard_ajax": "1",
                "s": str(params.s),
                "account": str(params.account),
                "password": enc_pwd,
                "rsatimestamp": ts,
            }
        )
        logger.debug(f"[AutoSteamRent] DoChangePassword: {r}")
        if r.get("errorMsg"):
            raise Exception(f"ChangePassword error: {r['errorMsg']}")
        if not r.get("success") and not r.get("hash"):
            raise Exception(f"ChangePassword no success: {r}")

    @staticmethod
    def _encrypt(password: str, mod: str, exp: str) -> str:
        pk = rsa.PublicKey(n=int(mod, 16), e=int(exp, 16))
        return base64.b64encode(rsa.encrypt(password.encode("ascii"), pk)).decode()

async def change_password_async(mafile: dict, current_password: str) -> str:
    return await SteamPasswordChanger(mafile, current_password).change_password()

def change_password_sync(mafile: dict, current_password: str, acc_id: int = 0) -> str:
    lock = _get_acc_lock(acc_id) if acc_id else _pwd_change_lock
    with lock:
        result = [None]
        error = [None]
        loop_ref = [None]
        main_task_ref = [None]
        done_evt = threading.Event()
        async def _runner():
            main_task_ref[0] = asyncio.current_task()
            return await change_password_async(mafile, current_password)
        def _run():
            loop = asyncio.new_event_loop()
            loop_ref[0] = loop
            asyncio.set_event_loop(loop)
            try:
                result[0] = loop.run_until_complete(_runner())
            except asyncio.CancelledError as e:
                error[0] = Exception("Password change cancelled (timeout)")
            except Exception as e:
                error[0] = e
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
                done_evt.set()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        if not done_evt.wait(timeout=PASSWORD_CHANGE_TIMEOUT):
            loop = loop_ref[0]
            task = main_task_ref[0]
            if loop and task:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except Exception:
                    pass
            done_evt.wait(timeout=15)
            if t.is_alive():
                raise Exception("Password change timed out (worker did not stop)")
            raise Exception("Password change timed out")
        if error[0]:
            raise error[0]
        if result[0] is None:
            raise Exception("Password change returned no result")
        return result[0]

class LotConfig(BaseModel):
    tag: str
    hours: int
    class Config:
        extra = "allow"

class MessagesConfig(BaseModel):
    order_completed: str  = ("✅ Данные от аккаунта:\n∟ Логин: $login\n∟ Пароль: $password\n"
                             "∟ Аренда на: $rent_period\n\n⚠️ Для входа вам понадобится Steam Guard код.\n"
                             "Напишите !code чтобы получить код")
    guard_code: str       = "✅ Steam Guard код: $code\n∟ Действителен ~30 секунд\n∟ Аренда до: $end_time"
    rent_over: str        = "⛔ Аренда завершена!\n∟ Пароль изменён"
    warning: str          = "⚠️ Аренда заканчивается через 10 минут!"
    extended: str         = "✅ Аренда продлена на +$hours ч.\n∟ Окончание: $end_time"
    auto_extended: str    = "✅ Аренда автоматически продлена на +$hours ч.\n∟ Окончание: $end_time"
    bonus: str            = "✅ Бонус за отзыв: +$hours ч."
    time_info: str        = "✅ Осталось: $remaining\n∟ Окончание: $end_time"
    error_msg: str        = "❌ Произошла ошибка! Ожидайте ответа продавца"
    no_accounts: str      = "❌ Нет свободных аккаунтов! Средства будут возвращены"
    refunded: str         = "✅ Средства возвращены"
    rent_expired: str     = "⛔ Время аренды истекло!"
    no_order: str         = "❌ Активный заказ не найден"
    no_account: str       = "❌ Аккаунт не найден"
    code_error: str       = "❌ Ошибка генерации кода, попробуйте через 30 сек"
    config_error: str     = "❌ Ошибка конфигурации, обратитесь к продавцу"
    rent_not_started: str = "⚠️ Напишите !code чтобы начать аренду"
    extend_link: str      = "🔄 Для продления аренды оплатите лот по ссылке:\n$link\n\n∟ Осталось: $remaining"
    extend_no_lot: str    = "❌ Лот для продления не найден, обратитесь к продавцу"
    stock_info: str       = "📦 Доступно для аренды:\n$stock_list"
    stock_empty: str      = "❌ Сейчас нет доступных аккаунтов"
    DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "order_completed":  "📋 Выдача данных",
        "guard_code":       "🔑 Steam Guard код",
        "rent_over":        "⛔ Конец аренды",
        "warning":          "⚠️ Предупреждение 10 мин",
        "extended":         "✅ Продление",
        "auto_extended":    "🔄 Авто-продление",
        "bonus":            "🎁 Бонус за отзыв",
        "time_info":        "⏱ Команда !time",
        "rent_expired":     "⏰ Время истекло",
        "error_msg":        "❌ Общая ошибка",
        "no_accounts":      "📭 Нет аккаунтов",
        "refunded":         "💰 Возврат",
        "no_order":         "🔍 Заказ не найден",
        "no_account":       "👤 Аккаунт не найден",
        "code_error":       "❌ Ошибка кода",
        "config_error":     "⚙️ Ошибка конфигурации",
        "rent_not_started": "⏳ Аренда не начата",
        "extend_link":      "🔗 Ссылка на продление",
        "extend_no_lot":    "❌ Лот не найден",
        "stock_info":       "📦 Наличие",
        "stock_empty":      "📭 Нет аккаунтов (склад)",
    }
    class Config:
        extra = "allow"

class ReviewRule(BaseModel):
    rent_hours: int
    bonus_hours: float
    class Config:
        extra = "allow"

class AccountModel(BaseModel):
    id: int
    login: str
    password: str
    mafile: Dict[str, Any]
    tag: str = "default"
    allowed_hours: List[int] = [24]
    status: str = RentStatus.FREE
    current_order: Optional[str] = None
    rental_end: Optional[str] = None
    owner: Optional[str] = None
    owner_id: Optional[int] = None
    owner_chat_id: Optional[Any] = None
    rental_start: Optional[str] = None
    rent_hours: int = 24
    access_count: int = 0
    class Config:
        extra = "allow"

@dataclass
class RentOrder:
    id: str
    chat_id: Optional[int]
    buyer: str
    buyer_id: int
    acc_id: int
    acc_login: str
    acc_tag: str
    hours: float
    status: str = RentStatus.BUSY
    warned: bool = False
    review_claimed: bool = False
    created_at: str = field(default_factory=lambda: _fmt(_now()))
    is_extension: bool = False
    lot_id: Optional[str] = None

    def __post_init__(self):
        if self.chat_id is not None:
            try:
                self.chat_id = int(self.chat_id)
            except (TypeError, ValueError):
                self.chat_id = None

    def update(self, **kwargs):
        with _data_lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            try:
                _save_orders()
            except Exception as _e:
                logger.debug(f"[AutoSteamRent] RentOrder.update save failed: {_e}")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "chat_id", "buyer", "buyer_id", "acc_id", "acc_login", "acc_tag",
            "hours", "status", "warned", "review_claimed", "created_at",
            "is_extension", "lot_id")}

class Settings(BaseModel):
    enabled: bool = False
    autoback_on_error: bool = False
    auto_extend: bool = False
    auto_disable_lots: bool = False
    auto_enable_lots: bool = False
    lots: Dict[str, Any] = {}
    review_rules: List[Dict[str, Any]] = [
        {"rent_hours": 3, "bonus_hours": 1.0}, {"rent_hours": 6, "bonus_hours": 2.0},
        {"rent_hours": 12, "bonus_hours": 4.0}, {"rent_hours": 24, "bonus_hours": 6.0},
        {"rent_hours": 72, "bonus_hours": 12.0}, {"rent_hours": 168, "bonus_hours": 24.0},
    ]
    messages: MessagesConfig = MessagesConfig()
    notification_order_completed: bool = True
    notification_error: bool = True
    notification_refund: bool = True
    class Config:
        extra = "allow"

    def toggle(self, p):
        setattr(self, p, not getattr(self, p))
        _save_settings()

    def set_message(self, k, v):
        setattr(self.messages, k, v)
        _save_settings()

    def has_lot(self, lot_id) -> bool:
        return str(lot_id) in self.lots

    def get_lot(self, lot_id: str) -> Optional[LotConfig]:
        raw = self.lots.get(str(lot_id))
        if raw is None:
            return None
        if isinstance(raw, str):
            return LotConfig(tag=_ntag(raw), hours=24)
        if isinstance(raw, dict):
            return LotConfig(tag=_ntag(raw.get("tag", "default")), hours=raw.get("hours", 24))
        return None

    def set_lot(self, lot_id: str, tag: str, hours: int):
        self.lots[str(lot_id)] = {"tag": _ntag(tag), "hours": hours}
        _save_settings()

    def del_lot(self, lot_id: str):
        self.lots.pop(str(lot_id), None)
        _save_settings()

    def rename_lot(self, old_id: str, new_id: str) -> bool:
        old_id, new_id = str(old_id), str(new_id)
        if old_id not in self.lots or new_id == old_id:
            return False
        self.lots[new_id] = self.lots.pop(old_id)
        _save_settings()
        return True

    def find_lot_id_by_tag(self, tag: str) -> Optional[str]:
        tag = _ntag(tag)
        for lid in self.lots:
            lc = self.get_lot(lid)
            if lc and _ntag(lc.tag) == tag:
                return lid
        return None

    def get_review_rules(self) -> List[ReviewRule]:
        return sorted([ReviewRule(**r) for r in self.review_rules if isinstance(r, dict)],
                      key=lambda x: x.rent_hours)

    def add_review_rule(self, rent_hours: int, bonus_hours: float):
        self.review_rules = [r for r in self.review_rules
                             if not (isinstance(r, dict) and r.get("rent_hours") == rent_hours)]
        self.review_rules.append({"rent_hours": rent_hours, "bonus_hours": bonus_hours})
        _save_settings()

    def del_review_rule(self, rent_hours: int):
        self.review_rules = [r for r in self.review_rules
                             if not (isinstance(r, dict) and r.get("rent_hours") == rent_hours)]
        _save_settings()

    def get_bonus_for_hours(self, hours: float) -> float:
        bonus = 0.0
        for rule in self.get_review_rules():
            if hours >= rule.rent_hours:
                bonus = rule.bonus_hours
        return bonus

SETTINGS: Optional[Settings] = None

ACCOUNTS: List[AccountModel] = []

ORDERS: Dict[str, RentOrder] = {}

cardinal_ref: Optional[Cardinal] = None

tg_logs: Optional[Any] = None

_code_cooldowns: Dict[str, float] = {}

_cooldowns_lock = threading.Lock()

_processed_orders: Set[str] = set()

class _TTLDict(dict):
    TTL_SECONDS = 1800

    def __init__(self):
        super().__init__()
        self._ts: Dict[int, float] = {}
        self._lock = threading.Lock()

    def _touch(self, key):
        self._ts[key] = time.time()

    def _expire(self):
        now = time.time()
        with self._lock:
            stale = [k for k, ts in list(self._ts.items()) if now - ts > self.TTL_SECONDS]
            for k in stale:
                self._ts.pop(k, None)
                dict.pop(self, k, None)

    def __getitem__(self, key):
        with self._lock:
            v = dict.__getitem__(self, key)
            self._touch(key)
        return v

    def __setitem__(self, key, value):
        self._expire()
        with self._lock:
            dict.__setitem__(self, key, value)
            self._touch(key)

    def get(self, key, default=None):
        with self._lock:
            if key in self:
                self._touch(key)
                return dict.__getitem__(self, key)
        return default

    def setdefault(self, key, default=None):
        self._expire()
        with self._lock:
            if key not in self:
                dict.__setitem__(self, key, default)
            self._touch(key)
            return dict.__getitem__(self, key)

    def pop(self, key, *args):
        with self._lock:
            self._ts.pop(key, None)
            return dict.pop(self, key, *args)

_temp_storage: Dict[int, dict] = _TTLDict()

_last_assigned: Dict[str, int] = {}

_data_lock = threading.Lock()

_processed_lock = threading.Lock()

_toggling_tags: Set[str] = set()
_toggling_lock = threading.Lock()

_stop_event = threading.Event()

def _save_settings():
    _save_json("settings", SETTINGS.dict())

def _save_accounts():
    _save_json("accounts", [a.dict() for a in ACCOUNTS])

def _save_orders():
    _save_json("orders", {k: v.to_dict() for k, v in ORDERS.items()})

def _cleanup_orders():
    removed = []
    with _data_lock:
        if len(ORDERS) <= MAX_ORDERS_STORED:
            return
        cutoff_dt = _now() - timedelta(days=ORDERS_MAX_AGE_DAYS)
        cutoff = _fmt(cutoff_dt)
        to_remove = [k for k, o in ORDERS.items()
                     if o.status in (RentStatus.FINISHED, RentStatus.REFUND) and o.created_at < cutoff]
        for k in to_remove:
            del ORDERS[k]
            removed.append(k)
        if len(ORDERS) > MAX_ORDERS_STORED:
            finished = sorted(
                [(k, o) for k, o in ORDERS.items() if o.status in (RentStatus.FINISHED, RentStatus.REFUND)],
                key=lambda x: x[1].created_at)
            while len(ORDERS) > MAX_ORDERS_STORED and finished:
                k, _ = finished.pop(0)
                del ORDERS[k]
                removed.append(k)
        _save_orders()
    if removed:
        with _processed_lock:
            for k in removed:
                _processed_orders.discard(k)

def _cleanup_processed():
    with _processed_lock:
        if len(_processed_orders) > MAX_PROCESSED_IDS:
            _processed_orders.clear()
            _processed_orders.update(ORDERS.keys())

def _cleanup_cooldowns():
    now_ts = time.time()
    with _cooldowns_lock:
        stale = [k for k, v in _code_cooldowns.items() if now_ts - v > CODE_COOLDOWN * 6]
        for k in stale:
            del _code_cooldowns[k]

def _cleanup_temp():
    _temp_storage._expire()
    if len(_temp_storage) > 200:
        _temp_storage.clear()

def _load_all():
    global SETTINGS, ACCOUNTS, ORDERS
    raw = _load_json("settings")
    if "review_rules" in raw and isinstance(raw["review_rules"], dict):
        raw["review_rules"] = [{"rent_hours": int(k), "bonus_hours": v}
                                for k, v in raw["review_rules"].items()]
    SETTINGS = Settings(**raw)
    changed = False
    for lid, val in list(SETTINGS.lots.items()):
        if isinstance(val, str):
            SETTINGS.lots[lid] = {"tag": _ntag(val), "hours": 24}
            changed = True
        elif isinstance(val, dict):
            val.pop("count", None)
            if "tag" not in val:
                val["tag"] = "default"
            if "hours" not in val:
                val["hours"] = 24
            changed = True
    if changed:
        _save_settings()
    d = _load_json("accounts")
    if isinstance(d, list):
        for a in d:
            if "rent_hours" in a and "allowed_hours" not in a:
                a["allowed_hours"] = [a["rent_hours"]]
        ACCOUNTS = [AccountModel(**a) for a in d]
    else:
        ACCOUNTS = []
    d = _load_json("orders")
    if isinstance(d, dict):
        for k, v in d.items():
            v.pop("acc_ids", None)
            v.pop("is_multi", None)
            v.setdefault("is_extension", False)
            v.setdefault("lot_id", None)
            v.setdefault("acc_login", "")
            v.setdefault("acc_tag", "")
        ORDERS = {k: RentOrder(**v) for k, v in d.items()}
    else:
        ORDERS = {}
    with _processed_lock:
        _processed_orders.update(ORDERS.keys())
    _cleanup_orders()

_load_all()

class AccountRepo:

    @staticmethod
    def get(acc_id: int) -> Optional[AccountModel]:
        return next((a for a in ACCOUNTS if a.id == acc_id), None)

    @staticmethod
    def by_order(order_id: str) -> Optional[AccountModel]:
        return next((a for a in ACCOUNTS if a.current_order == order_id), None)

    @staticmethod
    def get_free(tag: str, hours: int = None) -> Optional[AccountModel]:
        tag = _ntag(tag)
        with _data_lock:
            candidates = [
                a for a in ACCOUNTS
                if _ntag(a.tag) == tag
                and a.status == RentStatus.FREE
                and (hours is None or hours in a.allowed_hours)
            ]
            if not candidates:
                return None
            last_id = _last_assigned.get(tag, 0)
            candidates_after = [a for a in candidates if a.id > last_id]
            chosen = candidates_after[0] if candidates_after else candidates[0]
            _last_assigned[tag] = chosen.id
            return chosen

    @staticmethod
    def count_free(tag: str = None) -> Dict[str, int]:
        result = {}
        with _data_lock:
            snapshot = list(ACCOUNTS)
        for a in snapshot:
            if a.status != RentStatus.FREE:
                continue
            t = _ntag(a.tag)
            if tag is not None and t != _ntag(tag):
                continue
            result[t] = result.get(t, 0) + 1
        return result

    @staticmethod
    def get_free_periods(tag: str) -> List[int]:
        tag = _ntag(tag)
        periods = set()
        for a in ACCOUNTS:
            if _ntag(a.tag) == tag and a.status == RentStatus.FREE:
                periods.update(a.allowed_hours)
        return sorted(periods)

    @staticmethod
    def claim_free(tag, hours, order_id, buyer, buyer_id, chat_id) -> Optional[AccountModel]:
        tag_n = _ntag(tag)
        with _data_lock:
            candidates = [
                a for a in ACCOUNTS
                if _ntag(a.tag) == tag_n
                and a.status == RentStatus.FREE
                and (hours is None or hours in a.allowed_hours)
            ]
            if not candidates:
                return None
            last_id = _last_assigned.get(tag_n, 0)
            after = [a for a in candidates if a.id > last_id]
            chosen = after[0] if after else candidates[0]
            _last_assigned[tag_n] = chosen.id
            chosen.status = RentStatus.BUSY
            chosen.current_order = order_id
            chosen.owner = buyer
            chosen.owner_id = buyer_id
            chosen.owner_chat_id = chat_id
            chosen.rental_start = _fmt(_now())
            chosen.rental_end = None
            chosen.rent_hours = hours
            _save_accounts()
            return chosen

    @staticmethod
    def add(login, password, mafile, tag, allowed_hours=None) -> Tuple[bool, str]:
        if allowed_hours is None:
            allowed_hours = [24]
        tag = _ntag(tag)
        with _data_lock:
            if any(a.login.lower() == login.lower() for a in ACCOUNTS):
                return False, "Аккаунт уже существует"
            nid = max((a.id for a in ACCOUNTS), default=0) + 1
            ACCOUNTS.append(AccountModel(
                id=nid, login=login, password=password, mafile=mafile,
                tag=tag, allowed_hours=sorted(allowed_hours), rent_hours=allowed_hours[0]))
            _save_accounts()
        return True, f"Аккаунт {login} добавлен (ID: {nid}, тег: {tag}, периоды: {_format_periods(allowed_hours)})"

    @staticmethod
    def delete(acc_id: int) -> bool:
        with _data_lock:
            for i, a in enumerate(ACCOUNTS):
                if a.id == acc_id:
                    del ACCOUNTS[i]
                    _save_accounts()
                    with _acc_pwd_locks_mutex:
                        _acc_pwd_locks.pop(acc_id, None)
                    return True
        return False

    @staticmethod
    def assign(acc_id, order_id, buyer, buyer_id, chat_id, hours):
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return
            acc.status = RentStatus.BUSY
            acc.current_order = order_id
            acc.owner = buyer
            acc.owner_id = buyer_id
            acc.owner_chat_id = chat_id
            acc.rental_start = _fmt(_now())
            acc.rental_end = None
            acc.rent_hours = hours
            _save_accounts()

    @staticmethod
    def start_rent(order_id) -> Optional[AccountModel]:
        with _data_lock:
            acc = AccountRepo.by_order(order_id)
            if acc:
                acc.status = RentStatus.ACTIVE
                acc.rental_end = _fmt(_now() + timedelta(hours=acc.rent_hours))
                _save_accounts()
            return acc

    @staticmethod
    def extend_rent(acc_id: int, hours: float) -> Optional[str]:
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if acc and acc.rental_end:
                acc.rental_end = _fmt(_parse(acc.rental_end) + timedelta(hours=hours))
                _save_accounts()
                return acc.rental_end
        return None

    @staticmethod
    def release(acc_id: int, new_password: str = None, error: bool = False):
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return
            acc.status = RentStatus.ERROR if error else RentStatus.FREE
            acc.current_order = acc.owner = acc.owner_id = None
            acc.owner_chat_id = acc.rental_start = acc.rental_end = None
            acc.access_count = 0
            if new_password:
                acc.password = new_password
            _save_accounts()

    @staticmethod
    def reset_to_free(acc_id: int):
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return
            acc.status = RentStatus.FREE
            acc.current_order = acc.owner = acc.owner_id = None
            acc.owner_chat_id = acc.rental_start = acc.rental_end = None
            acc.access_count = 0
            _save_accounts()

    @staticmethod
    def manual_assign(acc_id: int, buyer: str, hours: int) -> Optional[AccountModel]:
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc or acc.status not in (RentStatus.FREE, RentStatus.ERROR):
                return None
            oid = f"manual_{acc_id}_{int(time.time())}"
            now = _now()
            acc.status = RentStatus.ACTIVE
            acc.current_order = oid
            acc.owner = buyer
            acc.owner_id = acc.owner_chat_id = None
            acc.rental_start = _fmt(now)
            acc.rental_end = _fmt(now + timedelta(hours=hours))
            acc.rent_hours = hours
            acc.access_count = 0
            ORDERS[oid] = RentOrder(id=oid, chat_id=None, buyer=buyer, buyer_id=0,
                                    acc_id=acc.id, acc_login=acc.login, acc_tag=_ntag(acc.tag),
                                    hours=float(hours), status=RentStatus.ACTIVE)
            _save_accounts()
            _save_orders()
            return acc

    @staticmethod
    def update_allowed_hours(acc_id: int, allowed_hours: List[int]) -> bool:
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return False
            acc.allowed_hours = sorted(allowed_hours)
            if acc.status == RentStatus.FREE:
                acc.rent_hours = allowed_hours[0] if allowed_hours else 24
            _save_accounts()
            return True

    @staticmethod
    def set_password(acc_id: int, new_password: str) -> bool:
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return False
            acc.password = new_password
            _save_accounts()
            return True

    @staticmethod
    def set_mafile(acc_id: int, mafile: Dict[str, Any]) -> Tuple[bool, str]:
        with _data_lock:
            acc = AccountRepo.get(acc_id)
            if not acc:
                return False, "Аккаунт не найден"
            missing = _validate_mafile(mafile)
            if missing:
                return False, f"Отсутствует в maFile: {', '.join(missing)}"
            acc.mafile = mafile
            new_login = mafile.get("account_name")
            if isinstance(new_login, str) and new_login.strip():
                acc.login = new_login.strip()
            _save_accounts()
            return True, ""

    @staticmethod
    def get_stats() -> dict:
        r = {s: 0 for s in (RentStatus.FREE, RentStatus.ACTIVE, RentStatus.BUSY, RentStatus.ERROR)}
        for a in ACCOUNTS:
            if a.status in r:
                r[a.status] += 1
        r["total"] = len(ACCOUNTS)
        return r

    @staticmethod
    def all_tags() -> List[str]:
        return list({_ntag(a.tag) for a in ACCOUNTS})

    @staticmethod
    def find_active_by_buyer(buyer_id: int, tag: str = None) -> Optional[RentOrder]:
        for o in ORDERS.values():
            if o.status not in (RentStatus.ACTIVE, RentStatus.BUSY):
                continue
            if o.buyer_id == buyer_id:
                if tag is None:
                    return o
                acc = AccountRepo.get(o.acc_id)
                if acc and _ntag(acc.tag) == _ntag(tag):
                    return o
        return None

    @staticmethod
    def find_order_by_chat(chat_id, author_id=None, author_name=None) -> Optional[RentOrder]:
        key = str(chat_id)
        for o in ORDERS.values():
            if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                continue
            if str(o.chat_id or "") == key:
                return o
        if author_id and author_id > 0:
            for o in ORDERS.values():
                if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                    continue
                if o.buyer_id == author_id:
                    return o
        if author_name:
            al = author_name.strip().lower()
            for o in ORDERS.values():
                if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                    continue
                if o.buyer and o.buyer.strip().lower() == al:
                    return o
        if author_id and author_id > 0:
            for acc in ACCOUNTS:
                if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY) and acc.owner_id == author_id:
                    if acc.current_order and acc.current_order in ORDERS:
                        return ORDERS[acc.current_order]
        for acc in ACCOUNTS:
            if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY) and acc.owner_chat_id:
                if str(acc.owner_chat_id) == key:
                    if acc.current_order and acc.current_order in ORDERS:
                        return ORDERS[acc.current_order]
        return None

    @staticmethod
    def find_tag_by_chat(chat_id, author_id=None, author_name=None) -> Optional[str]:
        order = AccountRepo.find_order_by_chat(chat_id, author_id, author_name)
        if order:
            acc = AccountRepo.get(order.acc_id)
            if acc:
                return _ntag(acc.tag)
        if author_id and author_id > 0:
            for o in sorted(ORDERS.values(), key=lambda x: x.created_at, reverse=True):
                if o.buyer_id == author_id:
                    acc = AccountRepo.get(o.acc_id)
                    if acc:
                        return _ntag(acc.tag)
        return None

class TgLogs:

    def __init__(self, c: Cardinal):
        self.c = c
        self.bot = c.telegram.bot

    def _send(self, text):
        for uid in self.c.telegram.authorized_users:
            try:
                self.bot.send_message(uid, f"<b>--- Auto Steam Rent ---</b>\n{text}", parse_mode="HTML")
            except Exception:
                pass

    def order_completed(self, order, login):
        if SETTINGS.notification_order_completed:
            self._send(f"✅ Заказ #{order.id[:12]}...\n∟ Аккаунт: {login}\n∟ Покупатель: {order.buyer}")

    def error(self, msg):
        if SETTINGS.notification_error:
            self._send(f"❌ Ошибка: {msg}")

    def refund(self, order_id, reason):
        if SETTINGS.notification_refund:
            self._send(f"💰 Возврат #{order_id[:12]}...\n∟ Причина: {reason}")

    def lots_auto_disabled(self, tag: str, lot_ids: List[str]):
        self._send(
            f"🔴 Авто-выключение лотов\n"
            f"∟ Тег: <code>{tag}</code>\n"
            f"∟ Лоты: {', '.join(f'#{lid}' for lid in lot_ids)}"
        )

    def lots_auto_enabled(self, tag: str, lot_ids: List[str]):
        self._send(
            f"🟢 Авто-включение лотов\n"
            f"∟ Тег: <code>{tag}</code>\n"
            f"∟ Лоты: {', '.join(f'#{lid}' for lid in lot_ids)}"
        )

def _tmpl(template: str, **kw) -> str:
    r = template
    for k, v in kw.items():
        r = r.replace(f"${k}", str(v))
    return r

def _send_fp(c, chat_id, text):
    delays = (0, 3, 8)
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            c.send_message(chat_id, text)
            return
        except Exception as e:
            last_err = e
            es = str(e).lower()
            if not any(s in es for s in ("429", "flood", "too many", "rate")):
                logger.debug(f"[AutoSteamRent] send_message non-retriable: {e}")
                return
    logger.warning(f"[AutoSteamRent] send_message gave up after retries: {last_err}")

def _do_refund(c, order_id) -> bool:
    try:
        c.account.refund(order_id)
        return True
    except Exception as e:
        s = str(e).lower()
        return "уже" in s or "already" in s

def _extract_lot_id_from_html(html: str) -> Optional[str]:
    if not html:
        return None
    for pat in (r'/lots/[^"\']*offer[^"\']*[?&]id=(\d+)',
                r'href=["\'][^"\']*[?&]id=(\d+)',
                r'data-offer=["\'](\d+)',
                r'data-id=["\'](\d+)'):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None

def _find_lot_id_for_order(c, event):
    order = event.order
    for attr in ("offer_id", "lot_id"):
        v = getattr(order, attr, None)
        if v is not None:
            sv = str(v)
            if SETTINGS.has_lot(sv):
                return sv
    html = getattr(order, "html", None) or ""
    extracted = _extract_lot_id_from_html(html)
    if extracted and SETTINGS.has_lot(extracted):
        return extracted
    order_id = getattr(order, "id", None)
    if order_id:
        try:
            full = c.account.get_order(order_id)
            for attr in ("offer_id", "lot_id"):
                v = getattr(full, attr, None)
                if v is not None and SETTINGS.has_lot(str(v)):
                    return str(v)
            for attr in ("html", "description"):
                v = getattr(full, attr, None)
                if v:
                    ex = _extract_lot_id_from_html(str(v))
                    if ex and SETTINGS.has_lot(ex):
                        return ex
            sub = getattr(full, "subcategory", None)
            sub_id = getattr(sub, "id", None) if sub else None
            if sub_id is not None:
                for lid in SETTINGS.lots.keys():
                    cfg = SETTINGS.get_lot(lid)
                    if cfg and getattr(cfg, "subcategory_id", None) == sub_id:
                        return str(lid)
        except Exception as e:
            logger.debug(f"[AutoSteamRent] get_order({order_id}) fallback failed: {e}")
    description = getattr(order, "description", None) or ""
    if description:
        m = _match_lot_by_description(c, description)
        if m:
            return m
    if len(SETTINGS.lots) == 1:
        only = next(iter(SETTINGS.lots.keys()))
        logger.warning(
            f"[AutoSteamRent] order={order_id}: лот не определён через API/html/description, "
            f"но в настройках только один лот — использую {only}"
        )
        return only
    logger.warning(
        f"[AutoSteamRent] order={order_id}: не удалось определить lot_id "
        f"(description={description[:80]!r}, html_len={len(html)})"
    )
    return None

def _match_lot_by_description(c, description):
    if not description or not SETTINGS.lots:
        return None
    try:
        all_lots = _get_cached_lots(c)
    except Exception:
        return None
    our_lot_ids = set(SETTINGS.lots.keys())
    our_lots = [lot for lot in all_lots if str(lot.id) in our_lot_ids]
    if not our_lots:
        return None
    desc_clean = description.strip().lower()
    desc_parts = [p.strip() for p in desc_clean.split(',') if p.strip()]
    for lot in our_lots:
        lot_title = (getattr(lot, 'description', None) or getattr(lot, 'title', None) or '').strip().lower()
        if lot_title and desc_clean == lot_title:
            return str(lot.id)
    best_id, best_score = None, 0.0
    for lot in our_lots:
        lot_title = (getattr(lot, 'description', None) or getattr(lot, 'title', None) or '').strip().lower()
        if not lot_title:
            continue
        lot_parts = [p.strip() for p in lot_title.split(',') if p.strip()]
        if not lot_parts or not desc_parts:
            continue
        matching = sum(1 for dp in desc_parts if dp in lot_parts)
        if matching > 0:
            score = matching / max(len(desc_parts), len(lot_parts))
            if score > best_score:
                best_score = score
                best_id = str(lot.id)
    return best_id if best_score >= 0.8 else None

def _get_extend_lot_id(order: RentOrder) -> Optional[str]:
    if order.lot_id and SETTINGS.has_lot(order.lot_id):
        return order.lot_id
    acc = AccountRepo.get(order.acc_id)
    if acc:
        return SETTINGS.find_lot_id_by_tag(_ntag(acc.tag))
    return None

def _build_stock_message(tag: str = None) -> str:
    free_counts = AccountRepo.count_free(tag)
    if not free_counts:
        return SETTINGS.messages.stock_empty
    lines = []
    for t in sorted(free_counts.keys()):
        periods = AccountRepo.get_free_periods(t)
        lines.append(f"∟ {t}: {free_counts[t]} шт. ({_format_periods(periods) if periods else '—'})")
    return _tmpl(SETTINGS.messages.stock_info, stock_list="\n".join(lines))

def _recover_account(c, acc, order, reason):
    acc_tag = _ntag(acc.tag)
    was_last_free = SETTINGS.auto_enable_lots and cardinal_ref and \
        AccountRepo.count_free(acc_tag).get(acc_tag, 0) == 0
    try:
        np = change_password_sync(acc.mafile, acc.password, acc.id)
        AccountRepo.release(acc.id, np)
        if was_last_free:
            def _auto_enable_recover(tag=acc_tag):
                toggled = _toggle_fp_lots_for_tag(cardinal_ref, tag, True)
                if toggled and tg_logs:
                    tg_logs.lots_auto_enabled(tag, toggled)
            threading.Thread(target=_auto_enable_recover, daemon=True).start()
        if order:
            order.update(status=RentStatus.FINISHED)
            if reason == "TIME" and order.chat_id:
                _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.rent_over, id=order.id))
    except SteamEmailVerificationRequired as e:
        logger.error(f"[AutoSteamRent] Email-verification: {acc.login} — {e}")
        AccountRepo.release(acc.id, error=True)
        if tg_logs:
            tg_logs.error(f"⚠️ {acc.login}: Steam требует email-подтверждение recovery. "
                          f"Залогинься с IP бота и подтверди письмом.")
        if SETTINGS.autoback_on_error and order:
            if _do_refund(c, order.id):
                order.update(status=RentStatus.REFUND)
                if tg_logs:
                    tg_logs.refund(order.id, f"Email-verification required: {acc.login}")
        return
    except Exception as e:
        logger.error(f"[AutoSteamRent] Смена пароля не удалась: {acc.login} — {e}")
        AccountRepo.release(acc.id, error=True)
        if tg_logs:
            tg_logs.error(f"Смена пароля: {acc.login} - {_safe_err(e)}")
        if SETTINGS.autoback_on_error and order:
            if _do_refund(c, order.id):
                order.update(status=RentStatus.REFUND)
                if tg_logs:
                    tg_logs.refund(order.id, f"Ошибка смены пароля: {acc.login}")

def _stats_text() -> str:
    now = time.time()
    finished_all = [o for o in ORDERS.values() if o.status == RentStatus.FINISHED]
    refunds_all = sum(1 for o in ORDERS.values() if o.status == RentStatus.REFUND)
    exts_all = sum(1 for o in ORDERS.values() if o.is_extension)
    h_all = sum(o.hours for o in finished_all)

    def agg(ts):
        threshold = _fmt(MOSCOW_TZ.localize(datetime.fromtimestamp(ts)))
        arr = [o for o in finished_all if o.created_at >= threshold]
        return len(arr), sum(o.hours for o in arr)
    c_d, h_d = agg(now - 86400)
    c_w, h_w = agg(now - 604800)
    c_m, h_m = agg(now - 2592000)
    s = AccountRepo.get_stats()
    return (
        "📊 <b>Статистика</b>\n\n"
        f"Аккаунтов: {s['total']} | 🟢{s[RentStatus.FREE]} 👤{s[RentStatus.ACTIVE]} "
        f"⏳{s[RentStatus.BUSY]} ❌{s[RentStatus.ERROR]}\n\n"
        f"∟ Сегодня: <code>{c_d}</code> аренд | <code>{h_d:.0f}</code> ч\n"
        f"∟ Неделя: <code>{c_w}</code> аренд | <code>{h_w:.0f}</code> ч\n"
        f"∟ Месяц: <code>{c_m}</code> аренд | <code>{h_m:.0f}</code> ч\n"
        f"∟ Всего: <code>{len(finished_all)}</code> аренд | <code>{h_all:.0f}</code> ч\n\n"
        f"Возвратов: {refunds_all} | Продлений: {exts_all}"
    )

def _order_detail_text(order_id: str):
    o = ORDERS.get(order_id)
    if not o:
        return "❌ Заказ не найден", None
    status_map = {
        RentStatus.FINISHED: "✅ Завершён", RentStatus.REFUND: "💰 Возврат",
        RentStatus.ACTIVE: "👤 Активна", RentStatus.BUSY: "⏳ Ожидание",
        RentStatus.ERROR: "❌ Ошибка"
    }
    st = status_map.get(o.status, o.status)
    acc = AccountRepo.get(o.acc_id)
    acc_name = acc.login if acc else (o.acc_login or f"#{o.acc_id}")
    order_url = FUNPAY_ORDER_URL.format(o.id)
    txt = f"📋 <b>Заказ <a href='{order_url}'>#{o.id}</a></b>\n\n"
    txt += f"∟ Статус: <b>{st}</b>\n"
    txt += f"∟ Покупатель: <code>{o.buyer}</code>\n"
    txt += f"∟ Аккаунт: <code>{acc_name}</code>\n"
    if o.lot_id:
        txt += f"∟ Лот: <code>{o.lot_id}</code>\n"
    txt += f"∟ Тег: <code>{o.acc_tag or '—'}</code>\n"
    txt += f"∟ Период: <code>{_period_label(int(o.hours))}</code>\n"
    txt += f"∟ Создан: <code>{o.created_at[:19]}</code>\n"
    if o.is_extension:
        txt += "∟ Тип: 🔄 Продление\n"
    if acc and acc.rental_end and o.status == RentStatus.ACTIVE:
        txt += f"∟ Осталось: <code>{_remaining_str(acc.rental_end)}</code>\n"
    if o.chat_id:
        chat_url = FUNPAY_CHAT_URL.format(o.chat_id)
        txt += f"∟ Чат: <a href='{chat_url}'>Перейти</a>\n"
    return txt, o

def process_new_order(c, event):
    if not SETTINGS or not SETTINGS.enabled:
        return
    order = event.order
    if not order:
        return
    order_id = getattr(order, 'id', None)
    if not order_id:
        return
    with _processed_lock:
        if order_id in _processed_orders:
            return
        _processed_orders.add(order_id)
    if order_id in ORDERS:
        return
    _cleanup_processed()
    lot_id = _find_lot_id_for_order(c, event)
    if not lot_id:
        return
    lot_cfg = SETTINGS.get_lot(lot_id)
    if not lot_cfg:
        return
    tag = _ntag(lot_cfg.tag)
    hours = lot_cfg.hours if lot_cfg.hours > 0 else 24
    buyer = getattr(order, 'buyer_username', None) or getattr(order, 'buyer', 'Unknown')
    buyer_id = getattr(order, 'buyer_id', 0) or 0
    chat_id = getattr(order, 'chat_id', None) or getattr(order, 'node_id', 0)
    if SETTINGS.auto_extend:
        with _data_lock:
            existing = AccountRepo.find_active_by_buyer(buyer_id, tag)
            if existing and order_id not in ORDERS:
                acc = AccountRepo.get(existing.acc_id)
                if acc and acc.rental_end:
                    from datetime import timedelta
                    acc.rental_end = _fmt(_parse(acc.rental_end) + timedelta(hours=hours))
                    ne = acc.rental_end
                    _save_accounts()
                    ORDERS[order_id] = RentOrder(
                        id=order_id, chat_id=chat_id, buyer=buyer, buyer_id=buyer_id,
                        acc_id=acc.id, acc_login=acc.login, acc_tag=_ntag(acc.tag),
                        hours=float(hours), status=RentStatus.ACTIVE,
                        is_extension=True, lot_id=lot_id)
                    _save_orders()
                    _send_fp(c, chat_id, _tmpl(SETTINGS.messages.auto_extended,
                                               hours=str(hours), end_time=ne))
                    return
    acc = AccountRepo.claim_free(tag, hours, order_id, buyer, buyer_id, chat_id)
    if not acc:
        if SETTINGS.autoback_on_error:
            _send_fp(c, chat_id, SETTINGS.messages.no_accounts)
            if _do_refund(c, order_id):
                _send_fp(c, chat_id, SETTINGS.messages.refunded)
                if tg_logs:
                    tg_logs.refund(order_id, f"Нет аккаунтов (тег: {tag})")
        else:
            logger.warning(f"[AutoSteamRent] Нет свободных аккаунтов для заказа {order_id} (тег: {tag}), autoback выключен")
            if tg_logs:
                tg_logs.error(f"Нет аккаунтов для заказа #{order_id[:12]} (тег: {tag}), возврат не выполнен")
        if SETTINGS.auto_disable_lots:
            def _auto_disable(c=c, tag=tag):
                toggled = _toggle_fp_lots_for_tag(c, tag, False)
                if toggled and tg_logs:
                    tg_logs.lots_auto_disabled(tag, toggled)
            threading.Thread(target=_auto_disable, daemon=True).start()
        return
    with _data_lock:
        ro = RentOrder(id=order_id, chat_id=chat_id, buyer=buyer, buyer_id=buyer_id,
                       acc_id=acc.id, acc_login=acc.login, acc_tag=_ntag(acc.tag),
                       hours=float(hours), lot_id=lot_id)
        ORDERS[order_id] = ro
        _save_orders()
    _send_fp(c, chat_id, _tmpl(SETTINGS.messages.order_completed,
                                login=acc.login, password=acc.password, id=order_id,
                                rent_period=_period_label(hours)))
    if tg_logs:
        tg_logs.order_completed(ro, acc.login)
    if SETTINGS.auto_disable_lots:
        remaining = AccountRepo.count_free(tag).get(_ntag(tag), 0)
        if remaining == 0:
            def _auto_disable_after(c=c, tag=tag):
                toggled = _toggle_fp_lots_for_tag(c, tag, False)
                if toggled and tg_logs:
                    tg_logs.lots_auto_disabled(tag, toggled)
            threading.Thread(target=_auto_disable_after, daemon=True).start()

def process_message(c, event):
    if not SETTINGS or not SETTINGS.enabled:
        return
    msg = event.message
    if not msg or not msg.text:
        return
    if msg.author_id == 0:
        if msg.type == MessageTypes.NEW_FEEDBACK:
            _handle_feedback(c, msg)
        return
    fl = msg.text.strip().split('\n', 1)[0].strip().lower()
    is_code = fl in _CMD_CODE
    is_time = fl in _CMD_TIME
    is_extend = fl in _CMD_EXTEND
    is_stock = fl in _CMD_STOCK
    if not (is_code or is_time or is_extend or is_stock):
        return
    author_name = getattr(msg, 'author', None) or getattr(msg, 'author_username', None)
    author_id = getattr(msg, 'author_id', None) or 0
    if is_stock:
        tag = AccountRepo.find_tag_by_chat(msg.chat_id, author_id, author_name)
        _send_fp(c, msg.chat_id, _build_stock_message(tag))
        return
    order = AccountRepo.find_order_by_chat(msg.chat_id, author_id, author_name)
    if not order:
        _send_fp(c, msg.chat_id, SETTINGS.messages.no_order)
        return
    acc = AccountRepo.get(order.acc_id)
    if not acc:
        _send_fp(c, msg.chat_id, SETTINGS.messages.no_account)
        return
    if order.chat_id != msg.chat_id:
        order.update(chat_id=msg.chat_id)
    if acc.owner_chat_id != msg.chat_id:
        with _data_lock:
            acc.owner_chat_id = msg.chat_id
            _save_accounts()
    if is_code:
        cd_key = str(msg.chat_id)
        now_ts = time.time()
        with _cooldowns_lock:
            if _code_cooldowns.get(cd_key, 0) > now_ts - CODE_COOLDOWN:
                return
            _code_cooldowns[cd_key] = now_ts
        ss = acc.mafile.get("shared_secret", "")
        if not ss:
            _send_fp(c, msg.chat_id, SETTINGS.messages.config_error)
            return
        code = SteamGuard.code_sync(ss)
        if code in ("ERROR", "NO_SECRET"):
            _send_fp(c, msg.chat_id, SETTINGS.messages.code_error)
            return
        if order.status == RentStatus.BUSY:
            acc_after = AccountRepo.start_rent(order.id)
            if acc_after:
                order.update(status=RentStatus.ACTIVE)
                acc = acc_after
        _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.guard_code,
                                        code=code, end_time=acc.rental_end or "?"))
        with _data_lock:
            acc.access_count += 1
            _save_accounts()
    elif is_time:
        if not acc.rental_end:
            _send_fp(c, msg.chat_id, SETTINGS.messages.rent_not_started)
        elif (_parse(acc.rental_end) - _now()).total_seconds() <= 0:
            _send_fp(c, msg.chat_id, SETTINGS.messages.rent_expired)
        else:
            _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.time_info,
                                            remaining=_remaining_str(acc.rental_end),
                                            end_time=acc.rental_end))
    elif is_extend:
        lot_id = _get_extend_lot_id(order)
        if not lot_id:
            _send_fp(c, msg.chat_id, SETTINGS.messages.extend_no_lot)
            return
        link = FUNPAY_LOT_URL.format(lot_id=lot_id)
        remaining = _remaining_str(acc.rental_end) if acc.rental_end else "—"
        _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.extend_link, link=link, remaining=remaining))

def _handle_feedback(c, message):
    try:
        from FunPayAPI.common.utils import RegularExpressions
        oids = RegularExpressions().ORDER_ID.findall(message.text or "")
    except Exception:
        return
    if not oids:
        return
    oid = oids[0].replace("#", "")
    order = ORDERS.get(oid)
    if not order or order.review_claimed:
        return
    bonus = SETTINGS.get_bonus_for_hours(order.hours)
    if bonus > 0:
        ne = AccountRepo.extend_rent(order.acc_id, bonus)
        if ne:
            order.update(review_claimed=True)
            _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.bonus, hours=str(bonus)))

def process_order_status_changed(c, event):
    if not SETTINGS.enabled or event.order.status not in (OrderStatuses.CLOSED, OrderStatuses.REFUNDED):
        return
    order = ORDERS.get(event.order.id)
    if not order or order.status in (RentStatus.FINISHED, RentStatus.REFUND):
        return
    if event.order.status == OrderStatuses.REFUNDED:
        acc = AccountRepo.by_order(event.order.id) or AccountRepo.get(order.acc_id)
        if acc:
            with _recovering_lock:
                if acc.id in _recovering_accounts:
                    return
                _recovering_accounts.add(acc.id)
            def _do_refund_recover(a=acc, o=order):
                try:
                    _recover_account(c, a, o, "REFUND_EXT")
                finally:
                    with _recovering_lock:
                        _recovering_accounts.discard(a.id)
            threading.Thread(target=_do_refund_recover, daemon=True).start()
    elif event.order.status == OrderStatuses.CLOSED:
        if order.status == RentStatus.BUSY:
            acc = AccountRepo.by_order(event.order.id) or AccountRepo.get(order.acc_id)
            if acc:
                threading.Thread(
                    target=_recover_account, args=(c, acc, order, "CLOSED_EARLY"), daemon=True
                ).start()
                return
        order.update(status=RentStatus.FINISHED)

_recovering_accounts: Set[int] = set()

_recovering_lock = threading.Lock()

def rental_check_loop(c):
    cleanup_counter = 0
    while not _stop_event.is_set():
        try:
            now = _now()
            with _data_lock:
                accounts_snapshot = list(ACCOUNTS)
            for acc in accounts_snapshot:
                if _stop_event.is_set():
                    return
                with _data_lock:
                    acc_status = acc.status
                    acc_order_id = acc.current_order
                    acc_rental_end = acc.rental_end
                    acc_id = acc.id
                if acc_status not in (RentStatus.ACTIVE, RentStatus.BUSY) or not acc_order_id:
                    continue
                order = ORDERS.get(acc_order_id)
                if not order:
                    AccountRepo.release(acc_id)
                    continue
                if acc_rental_end:
                    rem = (_parse(acc_rental_end) - now).total_seconds()
                    if 0 < rem < 600 and not order.warned:
                        if order.chat_id:
                            _send_fp(c, order.chat_id, SETTINGS.messages.warning)
                        order.update(warned=True)
                    if rem <= 0:
                        with _recovering_lock:
                            if acc_id in _recovering_accounts:
                                continue
                            _recovering_accounts.add(acc_id)
                        with _data_lock:
                            acc_snapshot = AccountRepo.get(acc_id)
                        order_snapshot = order
                        def _do_recover(a=acc_snapshot, o=order_snapshot):
                            try:
                                _recover_account(c, a, o, "TIME")
                            finally:
                                with _recovering_lock:
                                    _recovering_accounts.discard(a.id)
                        threading.Thread(target=_do_recover, daemon=True).start()
        except Exception as e:
            logger.error(f"[AutoSteamRent] rental_check_loop ошибка: {e}")
        cleanup_counter += 1
        if cleanup_counter >= 10:
            cleanup_counter = 0
            _cleanup_orders()
            _cleanup_cooldowns()
            _cleanup_temp()
            _cleanup_processed()
        _stop_event.wait(60)

class CBT:
    SP = f'{_CBT.PLUGIN_SETTINGS}:{UUID}'
    MAIN = "asr_main"
    ACC_MENU = "asr_accs"
    ACC_ADD = "asr_add"
    ACC_DEL = "asr_del"
    ACC_DEL_CONFIRM = "asr_adlcf"
    ACC_DEL_YES = "asr_adlyes"
    ACC_DEL_NO = "asr_adlno"
    ACC_LIST = "asr_lst"
    ACC_DETAIL = "asr_det"
    ACC_CODE = "asr_code"
    ACC_STOP = "asr_stop"
    ACC_CHPWD = "asr_chpwd"
    ACC_EXTEND = "asr_ext"
    ACC_EXTEND_DO = "asr_extdo"
    ACC_MANUAL = "asr_man"
    ACC_MANUAL_HOURS = "asr_manhr"
    ACC_EDIT_HOURS = "asr_ehrs"
    ACC_TOGGLE_HOUR = "asr_thrs"
    ACC_SAVE_HOURS = "asr_shrs"
    ACC_RESET = "asr_rst"
    ACC_SET_PWD = "asr_setpwd"
    ACC_EDIT_MAFILE = "asr_editma"
    LOTS = "asr_lots"
    LOT_ADD = "asr_ladd"
    LOT_TAG = "asr_ltag"
    LOT_HRS = "asr_lhrs"
    LOT_DETAIL = "asr_ldet"
    LOT_EDIT = "asr_ledt"
    LOT_EDIT_TAG = "asr_letag"
    LOT_EDIT_HRS = "asr_lehrs"
    LOT_RENAME = "asr_lren"
    LOT_DEL_CONFIRM = "asr_ldlcf"
    LOT_DEL_YES = "asr_ldlyes"
    LOT_DEL_NO = "asr_ldlno"
    LOT_TOGGLE_FP = "asr_ltglfp"
    LOTS_DISABLE_ALL = "asr_ldisall"
    LOTS_ENABLE_ALL = "asr_lenall"
    LOTS_SETTINGS = "asr_lsets"
    REVS = "asr_revs"
    REV_ADD = "asr_radd"
    REV_DEL = "asr_rdel"
    REV_HRS = "asr_rhrs"
    REV_BON = "asr_rbon"
    NOTIFS = "asr_ntf"
    MSGS = "asr_msgs"
    MSG_EDIT = "asr_medt"
    STATS = "asr_stat"
    FULL_STATS = "asr_fstat"
    HIST = "asr_hist"
    HIST_DETAIL = "asr_hdet"
    TOGGLE = "asr_tgl"
    FILES = "asr_files"
    FILES_CONFIRM = "asr_files_yes"
    HRS_TGL = "asr_htgl"
    HRS_DONE = "asr_hdone"

class States:
    LOGIN = "ASR_LOGIN"
    PASS = "ASR_PASS"
    TAG = "ASR_TAG"
    MAFILE = "ASR_MAFILE"
    MAN_BUYER = "ASR_MAN_BUYER"
    LOT_ID = "ASR_LOT_ID"
    LOT_RENAME = "ASR_LOT_RENAME"
    MSG_EDIT = "ASR_MSG_EDIT"
    SET_PWD = "ASR_SET_PWD"
    EDIT_MAFILE = "ASR_MAFILE_EDIT"

def _startup_diagnostics():
    issues = []
    if not ACCOUNTS:
        issues.append("Аккаунты не добавлены")
    else:
        bad_mafile = [a.login for a in ACCOUNTS if _validate_mafile(a.mafile)]
        bad_pwd = [a.login for a in ACCOUNTS if _warn_mafile(a.mafile)]
        if bad_mafile:
            issues.append(f"Неполный maFile (нет shared/identity_secret): {', '.join(bad_mafile[:3])}")
        if bad_pwd:
            issues.append(f"Нет device_id/SteamID (смена пароля недоступна): {', '.join(bad_pwd[:3])}")
    if not SETTINGS.lots:
        issues.append("Лоты не настроены")
    try:
        from playwright.sync_api import sync_playwright as _spw
        with _spw() as _p:
            if not os.path.exists(_p.chromium.executable_path):
                issues.append("Chromium не установлен — смена пароля может не работать")
    except Exception as e:
        issues.append(f"Playwright недоступен: {e} — смена пароля может не работать")
    if issues:
        logger.warning("[AutoSteamRent] Диагностика при старте:\n" +
                       "\n".join(f"  ⚠️ {i}" for i in issues))
    else:
        logger.info("[AutoSteamRent] Диагностика при старте: OK")

def init(card: Cardinal):
    global cardinal_ref, tg_logs
    cardinal_ref = card
    tg_logs = TgLogs(card)
    SteamGuard.sync_time_sync()
    _startup_diagnostics()
    if not card.telegram:
        threading.Thread(target=rental_check_loop, args=(card,), daemon=True).start()
        return
    tg, bot = card.telegram, card.telegram.bot

    def send(cid, text, kb=None):
        real_id = cid.chat.id if hasattr(cid, 'chat') else cid
        return bot.send_message(real_id, text, reply_markup=kb, parse_mode='HTML')

    def edit(msg_or_cb, text, kb=None):
        try:
            if hasattr(msg_or_cb, 'chat'):
                return bot.edit_message_text(text, msg_or_cb.chat.id, msg_or_cb.message_id,
                                             reply_markup=kb, parse_mode='HTML')
            elif hasattr(msg_or_cb, 'message'):
                return bot.edit_message_text(text, msg_or_cb.message.chat.id, msg_or_cb.message.message_id,
                                             reply_markup=kb, parse_mode='HTML')
        except Exception:
            pass

    def answer(cb, msg=None, alert=False):
        try:
            return bot.answer_callback_query(cb.id, msg, show_alert=alert)
        except Exception:
            pass

    def _p(c, idx=-1):
        return c.data.split(":")[idx]

    def _pid(c, idx=-1):
        return int(_p(c, idx))

    def _back_kb(cb=None):
        return K().add(B("⬅️ Назад", None, cb or CBT.MAIN))

    def _hours_kb(selected, toggle_cb, done_cb, back_cb):
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            check = "✅" if h in selected else "⬜"
            kb.add(B(f"{check} {_period_label(h)}", None, f"{toggle_cb}:{h}"))
        kb.add(B("✅ Готово", None, done_cb))
        kb.add(B("⬅️ Назад", None, back_cb))
        return kb

    def _try_delete(cid, mid):
        if not mid:
            return
        try:
            bot.delete_message(cid, mid)
        except Exception as e:
            err = str(e).lower()
            if "message to delete not found" in err:
                pass
            elif "message can't be deleted" in err:
                pass
            elif "bot was blocked" in err:
                pass
            else:
                logger.debug(f"[AutoSteamRent] delete chat={cid} mid={mid}: {e}")

    def _try_delete_many(cid, *message_ids):
        for mid in message_ids:
            _try_delete(cid, mid)

    def _ask(chat_id, user_id, state, text, kb=None):
        msg = bot.send_message(chat_id, text, reply_markup=kb, parse_mode='HTML')
        _temp_storage.setdefault(user_id, {})["bot_msg_id"] = msg.message_id
        tg.set_state(chat_id, msg.message_id, user_id, state, {})
        return msg.message_id

    def _cleanup_dialog(chat_id, user_id, user_msg_id):
        d = _temp_storage.get(user_id, {})
        bot_msg_id = d.get("bot_msg_id")
        tg.clear_state(chat_id, user_id, False)
        _try_delete_many(chat_id, user_msg_id, bot_msg_id)
        if "bot_msg_id" in d:
            del d["bot_msg_id"]

    def _main_text():
        s = AccountRepo.get_stats()
        active = sum(1 for o in ORDERS.values() if o.status in (RentStatus.ACTIVE, RentStatus.BUSY))
        return (f"<b>🎮 Настройки Auto Steam Rent</b>\n\n"
                f"∟ Аккаунтов: <code>{s['total']}</code>\n"
                f"∟ Лотов: <code>{len(SETTINGS.lots)}</code>\n"
                f"∟ Активных аренд: <code>{active}</code>\n")

    def _main_kb():
        kb = K(row_width=1)
        kb.row(B(f"{_is_on(SETTINGS.enabled)} Авто-выдача", None, f"{CBT.TOGGLE}:enabled"))
        kb.add(B(f"{_is_on(SETTINGS.autoback_on_error)} Авто-возврат", None, f"{CBT.TOGGLE}:autoback_on_error"))
        kb.add(B(f"{_is_on(SETTINGS.auto_extend)} Авто-продление", None, f"{CBT.TOGGLE}:auto_extend"))
        kb.add(B("📂 Аккаунты", None, CBT.ACC_MENU), B("🔗 Лоты", None, CBT.LOTS))
        kb.add(B("⭐️ Бонусы за отзывы", None, CBT.REVS))
        kb.row(B("🔔 Уведомления", None, CBT.NOTIFS), B("💬 Сообщения", None, CBT.MSGS))
        kb.row(B("📊 Статистика", None, CBT.STATS), B("📜 История", None, f"{CBT.HIST}:1"))
        kb.row(B("📁 Файлы", None, f"{CBT.FILES}:all"), B("⬅️ Назад", None, f"{_CBT.EDIT_PLUGIN}:{UUID}:0"))
        return kb

    def _acc_text(acc):
        icon = ICON_STATUS.get(acc.status, "❓")
        lines = [f"<b>{icon} Аккаунт #{acc.id}: {acc.login}</b>\n",
                 f"∟ Статус: <code>{acc.status}</code>",
                 f"∟ Тег: <code>{acc.tag}</code>",
                 f"∟ Периоды: <code>{_format_periods(acc.allowed_hours)}</code>",
                 f"∟ Пароль: <code>{acc.password}</code>"]
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            lines.append(f"∟ Аренда: <code>{_period_label(acc.rent_hours)}</code>")
            if acc.owner:
                lines.append(f"∟ Арендатор: <code>{acc.owner}</code>")
            if acc.rental_start:
                lines.append(f"∟ Начало: <code>{acc.rental_start}</code>")
            if acc.rental_end:
                lines.append(f"∟ Конец: <code>{acc.rental_end}</code>")
                lines.append(f"∟ Осталось: <code>{_remaining_str(acc.rental_end)}</code>")
            if acc.current_order:
                lines.append(f"∟ Заказ: <code>{acc.current_order[:20]}...</code>")
        lines.append(f"∟ Доступов: <code>{acc.access_count}</code>")
        return "\n".join(lines)

    def _acc_kb(acc):
        kb = K(row_width=2)
        kb.add(B("🔑 Выдать код", None, f"{CBT.ACC_CODE}:{acc.id}"),
               B("🔄 Сменить пароль", None, f"{CBT.ACC_CHPWD}:{acc.id}"))
        kb.add(B("✏️ Обновить пароль", None, f"{CBT.ACC_SET_PWD}:{acc.id}"),
               B("🗂 Обновить maFile", None, f"{CBT.ACC_EDIT_MAFILE}:{acc.id}"))
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            kb.add(B("⏹ Остановить", None, f"{CBT.ACC_STOP}:{acc.id}"),
                   B("⏰ Продлить", None, f"{CBT.ACC_EXTEND}:{acc.id}"))
        if acc.status in (RentStatus.FREE, RentStatus.ERROR):
            kb.add(B("🤝 Ручная аренда", None, f"{CBT.ACC_MANUAL}:{acc.id}"))
        if acc.status == RentStatus.ERROR:
            kb.add(B("🔓 Сброс FREE", None, f"{CBT.ACC_RESET}:{acc.id}"))
        kb.add(B("⏱ Периоды", None, f"{CBT.ACC_EDIT_HOURS}:{acc.id}"))
        kb.add(B("🗑 Удалить", None, f"{CBT.ACC_DEL_CONFIRM}:{acc.id}"))
        kb.add(B("⬅️ К списку", None, f"{CBT.ACC_LIST}:0"))
        return kb

    def open_main(c):
        edit(c.message, _main_text(), _main_kb())

    def open_main_cmd(m):
        send(m.chat.id, _main_text(), _main_kb())

    _TOGGLEABLE_SETTINGS = frozenset({
        "enabled", "autoback_on_error", "auto_extend", "auto_disable_lots",
        "auto_enable_lots", "notification_order_completed", "notification_error",
        "notification_refund",
    })

    def toggle_setting(c):
        p = _p(c)
        if p not in _TOGGLEABLE_SETTINGS:
            return answer(c, "❌ Недопустимое поле", True)
        SETTINGS.toggle(p)
        if p.startswith("notification"):
            open_notifs(c)
        elif p in ("auto_disable_lots", "auto_enable_lots"):
            open_lots(c)
        else:
            open_main(c)

    def open_acc_menu(c):
        kb = K()
        kb.add(B("➕ Добавить аккаунт", None, CBT.ACC_ADD))
        if ACCOUNTS:
            kb.add(B("📜 Список аккаунтов", None, f"{CBT.ACC_LIST}:0"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, "<b>📂 Управление аккаунтами</b>", kb)

    def open_acc_list(c):
        pg = _pid(c)
        kb = K(row_width=1)
        total = len(ACCOUNTS)
        tp = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        pg = max(0, min(pg, tp - 1))
        start, end = pg * PAGE_SIZE, (pg + 1) * PAGE_SIZE
        for acc in ACCOUNTS[start:end]:
            icon = ICON_STATUS.get(acc.status, "❓")
            owner = f' | {acc.owner}' if acc.owner else ''
            kb.add(B(f"{icon} {acc.login} [{acc.tag}] {_format_periods(acc.allowed_hours)}{owner}",
                     None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        nav = []
        if pg > 0:
            nav.append(B("⬅️", None, f"{CBT.ACC_LIST}:{pg - 1}"))
        nav.append(B(f"{pg + 1}/{tp}", None, _CBT.EMPTY))
        if end < total:
            nav.append(B("➡️", None, f"{CBT.ACC_LIST}:{pg + 1}"))
        if nav:
            kb.row(*nav)
        kb.add(B("⬅️ Назад", None, CBT.ACC_MENU))
        edit(c.message, f"<b>📜 Аккаунты ({total})</b>", kb)

    def open_acc_detail(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        edit(c.message, _acc_text(acc), _acc_kb(acc))

    def acc_del_confirm(c):
        aid = _pid(c)
        acc = AccountRepo.get(aid)
        if not acc:
            return answer(c, "❌ Не найден", True)
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            return answer(c, "❌ Аккаунт сейчас в аренде!", True)
        text = (
            f"⚠️ <b>Удалить аккаунт?</b>\n\n"
            f"∟ Логин: <code>{acc.login}</code>\n"
            f"∟ Тег: <code>{acc.tag}</code>\n"
            f"∟ Статус: <code>{acc.status}</code>\n\n"
            f"❗ Это действие необратимо!"
        )
        kb = K(row_width=2)
        kb.add(
            B("✅ Да", None, f"{CBT.ACC_DEL_YES}:{aid}"),
            B("❌ Нет", None, f"{CBT.ACC_DEL_NO}:{aid}")
        )
        edit(c.message, text, kb)

    def acc_del_yes(c):
        aid = _pid(c)
        acc = AccountRepo.get(aid)
        login = acc.login if acc else str(aid)
        AccountRepo.delete(aid)
        answer(c, f"✅ {login} удалён")
        c.data = f"{CBT.ACC_LIST}:0"
        open_acc_list(c)

    def acc_del_no(c):
        aid = _pid(c)
        answer(c, "❌ Удаление отменено")
        c.data = f"{CBT.ACC_DETAIL}:{aid}"
        open_acc_detail(c)

    def acc_code(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        ss = acc.mafile.get("shared_secret", "")
        if not ss:
            return answer(c, "❌ Нет shared_secret", True)
        code = SteamGuard.code_sync(ss)
        if code in ("ERROR", "NO_SECRET"):
            return answer(c, "❌ Ошибка генерации", True)
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY) and acc.owner_chat_id:
            _send_fp(card, acc.owner_chat_id,
                     _tmpl(SETTINGS.messages.guard_code, code=code, end_time=acc.rental_end or "?"))
        kb = K(row_width=2)
        kb.add(B("🔄 Новый код", None, f"{CBT.ACC_CODE}:{acc.id}"),
               B("⬅️ К аккаунту", None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        edit(c.message, f"🔑 <b>Steam Guard код</b>\n\n∟ Аккаунт: <code>{acc.login}</code>\n"
                        f"∟ Код: <code>{code}</code>\n∟ Действителен ~30 сек", kb)

    def acc_stop(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        if acc.status not in (RentStatus.ACTIVE, RentStatus.BUSY):
            return answer(c, "ℹ️ Не активна", True)
        with _recovering_lock:
            if acc.id in _recovering_accounts:
                return answer(c, "⏳ Уже идёт остановка", True)
            _recovering_accounts.add(acc.id)
        order = ORDERS.get(acc.current_order) if acc.current_order else None
        owner_chat_id = acc.owner_chat_id
        chat_id = c.message.chat.id
        acc_id = acc.id
        def _do():
            try:
                a = AccountRepo.get(acc_id)
                if a:
                    _recover_account(card, a, order, "MANUAL_STOP")
                    if owner_chat_id:
                        _send_fp(card, owner_chat_id, SETTINGS.messages.rent_over)
                    send(chat_id, f"✅ Аренда <code>{a.login}</code> остановлена.")
            except Exception as e:
                send(chat_id, f"❌ Ошибка остановки: {_safe_err(e)}")
            finally:
                with _recovering_lock:
                    _recovering_accounts.discard(acc_id)
        answer(c)
        edit(c.message, f"⏳ Остановка <code>{acc.login}</code>...", _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))
        threading.Thread(target=_do, daemon=True).start()

    def acc_chpwd(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        chat_id = c.message.chat.id
        acc_id = acc.id
        def _do():
            try:
                a = AccountRepo.get(acc_id)
                if not a:
                    send(chat_id, "❌ Аккаунт не найден")
                    return
                np = change_password_sync(a.mafile, a.password, a.id)
                with _data_lock:
                    a.password = np
                    _save_accounts()
                send(chat_id, f"✅ Пароль <code>{a.login}</code> изменён:\n<code>{np}</code>")
            except Exception as e:
                send(chat_id, f"❌ Ошибка: {_safe_err(e)}")
        answer(c)
        edit(c.message, f"⏳ Смена пароля <code>{acc.login}</code>...", _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))
        threading.Thread(target=_do, daemon=True).start()

    def acc_extend_menu(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        kb = K(row_width=3)
        for h in [1, 2, 3, 6, 12, 24]:
            kb.add(B(f"+{h}ч", None, f"{CBT.ACC_EXTEND_DO}:{acc.id}:{h}"))
        kb.add(B("⬅️", None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        edit(c.message, f"⏰ Продлить <code>{acc.login}</code>:", kb)

    def acc_extend_do(c):
        try:
            parts = c.data.split(":")
            aid, h = int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return answer(c, "❌ Неверные данные", True)
        ne = AccountRepo.extend_rent(aid, h)
        acc = AccountRepo.get(aid)
        if ne:
            if acc and acc.owner_chat_id:
                _send_fp(card, acc.owner_chat_id, _tmpl(SETTINGS.messages.extended, hours=str(h), end_time=ne))
            edit(c.message, f"✅ <code>{acc.login if acc else aid}</code> +{h}ч\n∟ Окончание: <code>{ne}</code>",
                 _back_kb(f"{CBT.ACC_DETAIL}:{aid}"))
        else:
            answer(c, "❌ Не удалось", True)

    def acc_reset(c):
        aid = _pid(c)
        acc = AccountRepo.get(aid)
        if not acc:
            return answer(c, "❌ Не найден", True)
        if acc.status != RentStatus.ERROR:
            return answer(c, "ℹ️ Не в ERROR", True)
        acc_tag = _ntag(acc.tag)
        if acc.current_order:
            order = ORDERS.get(acc.current_order)
            if order and order.status not in (RentStatus.FINISHED, RentStatus.REFUND):
                order.update(status=RentStatus.FINISHED)
        AccountRepo.reset_to_free(aid)
        answer(c, f"✅ {acc.login} → FREE")
        acc = AccountRepo.get(aid)
        edit(c.message, _acc_text(acc), _acc_kb(acc))
        if SETTINGS.auto_enable_lots and cardinal_ref:
            def _auto_enable_reset(tag=acc_tag):
                toggled = _toggle_fp_lots_for_tag(cardinal_ref, tag, True)
                if toggled and tg_logs:
                    tg_logs.lots_auto_enabled(tag, toggled)
            threading.Thread(target=_auto_enable_reset, daemon=True).start()

    def acc_edit_hours(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        _temp_storage[c.from_user.id] = {"ehrs_id": acc.id, "sel_hrs": list(acc.allowed_hours)}
        kb = _hours_kb(acc.allowed_hours, CBT.ACC_TOGGLE_HOUR,
                       f"{CBT.ACC_SAVE_HOURS}:{acc.id}", f"{CBT.ACC_DETAIL}:{acc.id}")
        edit(c.message, f"⏱ <b>Периоды для {acc.login}</b>\n\nВыберите (✅ = вкл):", kb)

    def acc_toggle_hour(c):
        h = _pid(c)
        d = _temp_storage.get(c.from_user.id, {})
        sel = d.get("sel_hrs", [])
        if h in sel:
            sel.remove(h)
        else:
            sel.append(h)
        aid = d.get("ehrs_id")
        acc = AccountRepo.get(aid) if aid else None
        kb = _hours_kb(sel, CBT.ACC_TOGGLE_HOUR, f"{CBT.ACC_SAVE_HOURS}:{aid}", f"{CBT.ACC_DETAIL}:{aid}")
        edit(c.message, f"⏱ <b>Периоды для {acc.login if acc else '?'}</b>\n\nВыберите (✅ = вкл):", kb)
        answer(c)

    def acc_save_hours(c):
        aid = _pid(c)
        sel = _temp_storage.get(c.from_user.id, {}).get("sel_hrs", [])
        if not sel:
            return answer(c, "❌ Выберите хотя бы один!", True)
        AccountRepo.update_allowed_hours(aid, sel)
        acc = AccountRepo.get(aid)
        if acc:
            edit(c.message, _acc_text(acc), _acc_kb(acc))
            answer(c, "✅ Сохранено!")
        else:
            answer(c, "❌ Не найден", True)

    def acc_set_pwd(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        _temp_storage.setdefault(c.from_user.id, {})["sp_acc_id"] = acc.id
        answer(c)
        _ask(c.message.chat.id, c.from_user.id,
             States.SET_PWD,
             f"✏️ Введите новый пароль для <code>{acc.login}</code>:",
             _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))

    def _h_set_pwd(m):
        d = _temp_storage.get(m.from_user.id, {})
        aid = d.get("sp_acc_id")
        pwd = (m.text or "").strip()
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        if not aid:
            send(m.chat.id, "❌ Данные утеряны, начните заново", _main_kb())
            return
        if not pwd:
            send(m.chat.id, "❌ Пароль не может быть пустым", _main_kb())
            return
        ok = AccountRepo.set_password(aid, pwd)
        acc = AccountRepo.get(aid)
        if ok and acc:
            send(m.chat.id, f"✅ Пароль для <code>{acc.login}</code> обновлён", _acc_kb(acc))
        else:
            send(m.chat.id, "❌ Не удалось обновить пароль", _main_kb())

    def acc_edit_mafile(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        _temp_storage.setdefault(c.from_user.id, {})["em_acc_id"] = acc.id
        _temp_storage[c.from_user.id]["em_current_login"] = acc.login
        answer(c)
        _ask(c.message.chat.id, c.from_user.id,
             States.EDIT_MAFILE,
             f"🗂 Отправьте <b>.maFile</b> для <code>{acc.login}</code> файлом или JSON текстом:",
             _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))

    def _read_mafile_content(m):
        if m.content_type == 'document' and m.document:
            file_info = bot.get_file(m.document.file_id)
            file_bytes = bot.download_file(file_info.file_path)
            return file_bytes.decode('utf-8')
        elif m.text:
            return m.text.strip()
        return None

    def _h_mafile_edit(m):
        if not tg.check_state(m.chat.id, m.from_user.id, States.EDIT_MAFILE):
            return
        d = _temp_storage.get(m.from_user.id, {})
        aid = d.get("em_acc_id")
        if not aid:
            _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
            send(m.chat.id, "❌ Данные утеряны, начните заново", _main_kb())
            return
        try:
            content = _read_mafile_content(m)
        except Exception as e:
            _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
            send(m.chat.id, f"❌ Ошибка чтения файла: {_safe_err(e)}", _main_kb())
            return
        if content is None:
            _cleanup_dialog(m.chat.id, m.from_user.id, None)
            send(m.chat.id, "❌ Отправьте .maFile файлом или JSON текстом", _main_kb())
            return
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        if not content.strip():
            send(m.chat.id, "❌ Файл пустой или не содержит текст", _main_kb())
            return
        try:
            mf = json.loads(content)
        except json.JSONDecodeError as e:
            send(m.chat.id, f"❌ Невалидный JSON: {_safe_err(e)}", _main_kb())
            return
        if not isinstance(mf, dict):
            send(m.chat.id, "❌ Неверный формат maFile. Ожидается JSON-объект.", _main_kb())
            return
        missing = _validate_mafile(mf)
        if missing:
            send(m.chat.id,
                 f"❌ В maFile отсутствуют обязательные поля:\n<code>{', '.join(missing)}</code>",
                 _main_kb())
            return
        current_login = d.get("em_current_login", "")
        mafile_login = mf.get("account_name", "").strip()
        ok, err = AccountRepo.set_mafile(aid, mf)
        acc = AccountRepo.get(aid)
        if ok and acc:
            extra = ""
            if mafile_login and current_login and mafile_login.lower() != current_login.lower():
                extra += f"\nℹ️ Логин обновлён: <code>{acc.login}</code>"
            warn = _warn_mafile(mf)
            if warn:
                extra += f"\n⚠️ Нет полей для смены пароля/деавторизации: <code>{', '.join(warn)}</code>"
            send(m.chat.id, f"✅ maFile для <code>{acc.login}</code> обновлён{extra}", _acc_kb(acc))
        else:
            send(m.chat.id, f"❌ {err or 'Не удалось обновить maFile'}", _main_kb())

    def acc_manual_start(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        if acc.status not in (RentStatus.FREE, RentStatus.ERROR):
            return answer(c, "ℹ️ Не свободен", True)
        _temp_storage.setdefault(c.from_user.id, {})["man_id"] = acc.id
        answer(c)
        _ask(c.message.chat.id, c.from_user.id,
             States.MAN_BUYER,
             f"🤝 Ручная аренда <code>{acc.login}</code>\n\nВведите <b>ник покупателя</b>:",
             _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))

    def _h_manual_buyer(m):
        _temp_storage.setdefault(m.from_user.id, {})["man_buyer"] = m.text.strip()
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.ACC_MANUAL_HOURS}:{h}"))
        send(m.chat.id, "Выберите <b>период</b>:", kb)

    def handle_manual_hours(c):
        h = _pid(c)
        d = _temp_storage.get(c.from_user.id, {})
        aid = d.get("man_id")
        if not aid:
            return answer(c, "❌ Начните заново", True)
        acc = AccountRepo.manual_assign(aid, d.get("man_buyer", "manual"), h)
        if acc:
            edit(c.message, f"✅ <code>{acc.login}</code> → <code>{acc.owner}</code> на {h}ч\n"
                            f"∟ Окончание: <code>{acc.rental_end}</code>",
                 _back_kb(f"{CBT.ACC_DETAIL}:{aid}"))
        else:
            edit(c.message, "❌ Не удалось (занят?)", _back_kb(f"{CBT.ACC_DETAIL}:{aid}"))

    def start_add(c):
        answer(c)
        _temp_storage[c.from_user.id] = {}
        _ask(c.message.chat.id, c.from_user.id,
             States.LOGIN,
             "1️⃣ Введите <b>логин</b>:",
             _back_kb(CBT.ACC_MENU))

    def _h_login(m):
        if m.text.startswith("/"):
            return
        login = m.text.strip()
        _temp_storage.setdefault(m.from_user.id, {})["login"] = login
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        _ask(m.chat.id, m.from_user.id,
             States.PASS,
             "2️⃣ Введите <b>пароль</b>:",
             _back_kb(CBT.ACC_MENU))

    def _h_pass(m):
        _temp_storage.setdefault(m.from_user.id, {})["password"] = m.text.strip()
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        _ask(m.chat.id, m.from_user.id,
             States.TAG,
             "3️⃣ Введите <b>тег</b>:",
             _back_kb(CBT.ACC_MENU))

    def _h_tag(m):
        d = _temp_storage.setdefault(m.from_user.id, {})
        d["tag"] = m.text.strip()
        d["sel_hrs"] = []
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        kb = _hours_kb([], CBT.HRS_TGL, CBT.HRS_DONE, CBT.ACC_MENU)
        msg = send(m.chat.id, "4️⃣ Выберите <b>периоды аренды</b>:", kb)
        _temp_storage[m.from_user.id]["bot_msg_id"] = msg.message_id

    def hrs_toggle(c):
        h = _pid(c)
        sel = _temp_storage.get(c.from_user.id, {}).get("sel_hrs", [])
        if h in sel:
            sel.remove(h)
        else:
            sel.append(h)
        kb = _hours_kb(sel, CBT.HRS_TGL, CBT.HRS_DONE, CBT.ACC_MENU)
        edit(c.message, "4️⃣ Выберите <b>периоды аренды</b>:", kb)
        answer(c)

    def hrs_done(c):
        d = _temp_storage.get(c.from_user.id, {})
        sel = d.get("sel_hrs", [])
        if not sel:
            return answer(c, "❌ Выберите хотя бы один!", True)
        d["allowed_hours"] = sorted(sel)
        answer(c)
        _try_delete(c.message.chat.id, c.message.message_id)
        _ask(c.message.chat.id, c.from_user.id,
             States.MAFILE,
             "5️⃣ Отправьте <b>.maFile</b> (файлом или JSON текстом):",
             _back_kb(CBT.ACC_MENU))

    def _h_mafile(m):
        if not tg.check_state(m.chat.id, m.from_user.id, States.MAFILE):
            return
        try:
            content = _read_mafile_content(m)
        except Exception as e:
            _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
            send(m.chat.id, f"❌ Ошибка чтения файла: {_safe_err(e)}", _main_kb())
            return
        if content is None:
            _cleanup_dialog(m.chat.id, m.from_user.id, None)
            send(m.chat.id, "❌ Отправьте .maFile файлом или JSON текстом", _main_kb())
            return
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        if not content.strip():
            send(m.chat.id, "❌ Файл пустой или не содержит текст", _main_kb())
            return
        try:
            mf = json.loads(content)
        except json.JSONDecodeError as e:
            send(m.chat.id, f"❌ Невалидный JSON: {_safe_err(e)}", _main_kb())
            return
        if not isinstance(mf, dict):
            send(m.chat.id, "❌ Неверный формат maFile. Ожидается JSON-объект.", _main_kb())
            return
        missing = _validate_mafile(mf)
        if missing:
            send(m.chat.id,
                 f"❌ В maFile отсутствуют обязательные поля:\n<code>{', '.join(missing)}</code>",
                 _main_kb())
            return
        d = _temp_storage.get(m.from_user.id, {})
        if "login" not in d:
            send(m.chat.id, "❌ Данные потеряны, начните заново", _main_kb())
            return
        mafile_login = mf.get("account_name", "").strip()
        entered_login = d["login"].strip()
        actual_login = mafile_login if mafile_login else entered_login
        ok, txt = AccountRepo.add(actual_login, d["password"], mf, d["tag"],
                                  d.get("allowed_hours", [24]))
        _invalidate_lots_cache()
        if ok:
            extra = ""
            if mafile_login and entered_login.lower() != mafile_login.lower():
                extra += f"\nℹ️ Использован логин из maFile: <code>{actual_login}</code>"
            warn = _warn_mafile(mf)
            if warn:
                extra += f"\n⚠️ Нет полей для смены пароля/деавторизации: <code>{', '.join(warn)}</code>"
            send(m.chat.id, f"✅ {txt}{extra}", _main_kb())
            if SETTINGS.auto_enable_lots and cardinal_ref:
                acc_tag = _ntag(d["tag"])
                def _auto_enable_add(tag=acc_tag):
                    toggled = _toggle_fp_lots_for_tag(cardinal_ref, tag, True)
                    if toggled and tg_logs:
                        tg_logs.lots_auto_enabled(tag, toggled)
                threading.Thread(target=_auto_enable_add, daemon=True).start()
        else:
            send(m.chat.id, f"❌ {txt}", _main_kb())

    def open_lots(c):
        fp_active: Dict[str, Optional[bool]] = {}
        try:
            fp_lots = _get_cached_lots(cardinal_ref)
            fp_active = {str(l.id): l.active for l in fp_lots}
            logger.debug(f"[AutoSteamRent] lots cache: {list(fp_active.keys())[:10]}, settings lots: {list(SETTINGS.lots.keys())[:10]}")
        except Exception as e:
            logger.warning(f"[AutoSteamRent] Не удалось получить статусы лотов с FunPay: {e}")
        count = len(SETTINGS.lots)
        kb = K(row_width=1)
        kb.row(B(f"{_is_on(SETTINGS.auto_disable_lots)} Авто-выкл при пустом складе",
                 None, f"{CBT.TOGGLE}:auto_disable_lots"))
        kb.row(B(f"{_is_on(SETTINGS.auto_enable_lots)} Авто-вкл при добавлении аккаунта",
                 None, f"{CBT.TOGGLE}:auto_enable_lots"))
        if count:
            kb.row(
                B("🔴 Выкл все", None, CBT.LOTS_DISABLE_ALL),
                B("🟢 Вкл все", None, CBT.LOTS_ENABLE_ALL),
            )
            for lid in SETTINGS.lots:
                lc = SETTINGS.get_lot(lid)
                if lc:
                    fp_status = fp_active.get(lid)
                    fp_icon = "🟢" if fp_status is True else ("🔴" if fp_status is False else "⚪")
                    free = AccountRepo.count_free(lc.tag).get(_ntag(lc.tag), 0)
                    kb.add(B(
                        f"{fp_icon} #{lid}  ·  {lc.tag}  ·  {_period_label(lc.hours)}  ·  {free} шт.",
                        None, f"{CBT.LOT_DETAIL}:{lid}"
                    ))
        kb.row(B("➕ Добавить лот", None, CBT.LOT_ADD))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        auto_dis = "🟢 вкл" if SETTINGS.auto_disable_lots else "🔴 выкл"
        auto_en  = "🟢 вкл" if SETTINGS.auto_enable_lots  else "🔴 выкл"
        if count:
            text = (
                f"<b>🔗 Лоты</b>\n\n"
                f"<b>Авто-управление</b>\n"
                f"∟ Выкл при пустом складе: {auto_dis}\n"
                f"∟ Вкл при добавлении аккаунта: {auto_en}\n\n"
                f"<b>Лоты</b> — всего: <code>{count}</code>\n"
                f"∟ 🟢 включён  🔴 выключен  ⚪ нет данных"
            )
        else:
            text = (
                f"<b>🔗 Лоты</b>\n\n"
                f"<b>Авто-управление</b>\n"
                f"∟ Выкл при пустом складе: {auto_dis}\n"
                f"∟ Вкл при добавлении аккаунта: {auto_en}\n\n"
                f"Лоты не добавлены."
            )
        edit(c.message, text, kb)

    def open_lot_detail(c):
        lid = _p(c)
        lc = SETTINGS.get_lot(lid)
        if not lc:
            return answer(c, "❌ Лот не найден", True)
        lot_url = FUNPAY_LOT_URL.format(lot_id=lid)
        free_count = AccountRepo.count_free(lc.tag).get(_ntag(lc.tag), 0)
        fp_active: Optional[bool] = None
        try:
            fp_lots = _get_cached_lots(cardinal_ref)
            fp_lot = next((l for l in fp_lots if str(l.id) == lid), None)
            if fp_lot is not None:
                fp_active = fp_lot.active
        except Exception:
            pass
        if fp_active is True:
            active_str = "🟢 Включён"
        elif fp_active is False:
            active_str = "🔴 Выключен"
        else:
            active_str = "⚪ Нет данных"
        text = (
            f"<b>🔗 Лот #{lid}</b>\n\n"
            f"∟ Тег: <code>{lc.tag}</code>\n"
            f"∟ Период: <code>{_period_label(lc.hours)}</code>\n"
            f"∟ Свободных аккаунтов: <code>{free_count}</code>\n"
            f"∟ Статус на FunPay: {active_str}\n"
            f"∟ Ссылка: {lot_url}"
        )
        kb = K(row_width=2)
        kb.add(
            B("✏️ Изменить тег", None, f"{CBT.LOT_EDIT}:{lid}"),
            B("🔢 Изменить ID", None, f"{CBT.LOT_RENAME}:{lid}")
        )
        if fp_active is True:
            kb.add(B("🔴 Выключить на FunPay", None, f"{CBT.LOT_TOGGLE_FP}:{lid}:0"))
        elif fp_active is False:
            kb.add(B("🟢 Включить на FunPay", None, f"{CBT.LOT_TOGGLE_FP}:{lid}:1"))
        else:
            kb.add(B("⚡ Вкл/Выкл на FunPay", None, f"{CBT.LOT_TOGGLE_FP}:{lid}:toggle"))
        kb.add(B("🗑 Удалить", None, f"{CBT.LOT_DEL_CONFIRM}:{lid}"))
        kb.add(B("⬅️ К списку", None, CBT.LOTS))
        edit(c.message, text, kb)

    def lot_rename(c):
        lid = _p(c)
        lc = SETTINGS.get_lot(lid)
        if not lc:
            return answer(c, "❌ Лот не найден", True)
        _temp_storage.setdefault(c.from_user.id, {})["rename_lot_old"] = lid
        answer(c)
        _ask(c.message.chat.id, c.from_user.id,
             States.LOT_RENAME,
             f"🔢 Текущий ID лота: <code>{lid}</code>\n\nВведите <b>новый ID лота</b>:",
             _back_kb(f"{CBT.LOT_DETAIL}:{lid}"))

    def _h_lot_rename(m):
        raw = (m.text or "").strip()
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        old_id = _temp_storage.get(m.from_user.id, {}).get("rename_lot_old")
        if not old_id:
            send(m.chat.id, "❌ Данные утеряны, начните заново", _main_kb())
            return
        new_id = _extract_lot_id(raw)
        if not new_id:
            send(m.chat.id,
                 "❌ Не удалось распознать ID лота. Введите число или ссылку funpay.com/lots/offer?id=...",
                 _back_kb(f"{CBT.LOT_DETAIL}:{old_id}"))
            return
        if SETTINGS.has_lot(new_id):
            send(m.chat.id, f"❌ Лот с ID <code>{new_id}</code> уже существует",
                 _back_kb(f"{CBT.LOT_DETAIL}:{old_id}"))
            return
        ok = SETTINGS.rename_lot(old_id, new_id)
        if ok:
            _invalidate_lots_cache()
            send(m.chat.id,
                 f"✅ ID лота изменён: <code>{old_id}</code> → <code>{new_id}</code>",
                 _back_kb(f"{CBT.LOT_DETAIL}:{new_id}"))
        else:
            send(m.chat.id, "❌ Не удалось изменить ID лота", _back_kb(f"{CBT.LOT_DETAIL}:{old_id}"))

    def lot_del_confirm(c):
        lid = _p(c)
        lc = SETTINGS.get_lot(lid)
        if not lc:
            return answer(c, "❌ Лот не найден", True)
        text = (
            f"⚠️ <b>Удалить лот?</b>\n\n"
            f"∟ ID: <code>{lid}</code>\n"
            f"∟ Тег: <code>{lc.tag}</code>\n"
            f"∟ Период: <code>{_period_label(lc.hours)}</code>\n\n"
            f"❗ Это действие необратимо!"
        )
        kb = K(row_width=2)
        kb.add(
            B("✅ Да", None, f"{CBT.LOT_DEL_YES}:{lid}"),
            B("❌ Нет", None, f"{CBT.LOT_DEL_NO}:{lid}")
        )
        edit(c.message, text, kb)

    def lot_del_yes(c):
        lid = _p(c)
        lc = SETTINGS.get_lot(lid)
        name = f"#{lid} ({lc.tag})" if lc else f"#{lid}"
        SETTINGS.del_lot(lid)
        _invalidate_lots_cache()
        answer(c, f"✅ Лот {name} удалён")
        open_lots(c)

    def lot_del_no(c):
        lid = _p(c)
        answer(c, "❌ Удаление отменено")
        c.data = f"{CBT.LOT_DETAIL}:{lid}"
        open_lot_detail(c)

    def lot_edit(c):
        lid = _p(c)
        lc = SETTINGS.get_lot(lid)
        if not lc:
            return answer(c, "❌ Лот не найден", True)
        _temp_storage.setdefault(c.from_user.id, {})["edit_lot_id"] = lid
        _temp_storage[c.from_user.id]["edit_lot_tag"] = lc.tag
        _temp_storage[c.from_user.id]["edit_lot_hrs"] = lc.hours
        tags = AccountRepo.all_tags()
        if not tags:
            return answer(c, "❌ Нет аккаунтов!", True)
        kb = K(row_width=2)
        for tag in tags:
            prefix = "✅ " if tag == lc.tag else ""
            kb.add(B(f"{prefix}{tag}", None, f"{CBT.LOT_EDIT_TAG}:{lid}:{tag}"))
        kb.add(B("⬅️ Назад", None, f"{CBT.LOT_DETAIL}:{lid}"))
        edit(c.message,
             f"✏️ <b>Изменить лот #{lid}</b>\n\n"
             f"Текущий тег: <code>{lc.tag}</code>\n\n"
             f"Выберите новый тег:", kb)

    def lot_edit_tag(c):
        try:
            parts = c.data.split(":")
            lid = parts[1]
            new_tag = _ntag(parts[2])
        except (IndexError, ValueError):
            return answer(c, "❌ Неверные данные", True)
        d = _temp_storage.setdefault(c.from_user.id, {})
        d["edit_lot_id"] = lid
        d["edit_lot_tag"] = new_tag
        lc = SETTINGS.get_lot(lid)
        cur_hrs = lc.hours if lc else 24
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            prefix = "✅ " if h == cur_hrs else ""
            kb.add(B(f"{prefix}{_period_label(h)}", None, f"{CBT.LOT_EDIT_HRS}:{lid}:{h}"))
        kb.add(B("⬅️ Назад", None, f"{CBT.LOT_EDIT}:{lid}"))
        edit(c.message,
             f"✏️ <b>Изменить лот #{lid}</b>\n\n"
             f"Тег: <code>{new_tag}</code>\n\n"
             f"Выберите период:", kb)

    def lot_edit_hrs(c):
        parts = c.data.split(":")
        lid = parts[1]
        new_hrs = int(parts[2])
        d = _temp_storage.get(c.from_user.id, {})
        new_tag = d.get("edit_lot_tag", "default")
        SETTINGS.set_lot(lid, new_tag, new_hrs)
        _invalidate_lots_cache()
        answer(c, "✅ Лот обновлён!")
        c.data = f"{CBT.LOT_DETAIL}:{lid}"
        open_lot_detail(c)

    def lot_toggle_fp(c):
        parts = c.data.split(":")
        lid = parts[1]
        action = parts[2] if len(parts) > 2 else "toggle"
        lc = SETTINGS.get_lot(lid)
        if not lc:
            return answer(c, "❌ Лот не найден", True)
        try:
            lf = cardinal_ref.account.get_lot_fields(int(lid))
            if action == "toggle":
                lf.active = not lf.active
            else:
                lf.active = bool(int(action))
            cardinal_ref.account.save_lot(lf)
            _invalidate_lots_cache()
            state = "🟢 включён" if lf.active else "🔴 выключен"
            answer(c, f"✅ Лот #{lid} {state}")
        except Exception as e:
            answer(c, f"❌ Ошибка: {_safe_err(e)}", True)
            return
        c.data = f"{CBT.LOT_DETAIL}:{lid}"
        open_lot_detail(c)

    def lots_disable_all(c):
        answer(c)
        edit(c.message, "⏳ Выключаю лоты на FunPay...", _back_kb(CBT.LOTS))
        chat_id = c.message.chat.id
        def _do():
            tags = list({_ntag((SETTINGS.get_lot(lid) or LotConfig(tag="default", hours=24)).tag)
                         for lid in SETTINGS.lots})
            total = []
            for tag in tags:
                total.extend(_toggle_fp_lots_for_tag(cardinal_ref, tag, False))
            if total and tg_logs:
                tg_logs.lots_auto_disabled("all", total)
            send(chat_id, f"🔴 Выключено лотов: {len(total)}" if total else "ℹ️ Нечего выключать")
        threading.Thread(target=_do, daemon=True).start()

    def lots_enable_all(c):
        answer(c)
        edit(c.message, "⏳ Включаю лоты на FunPay...", _back_kb(CBT.LOTS))
        chat_id = c.message.chat.id
        def _do():
            tags = list({_ntag((SETTINGS.get_lot(lid) or LotConfig(tag="default", hours=24)).tag)
                         for lid in SETTINGS.lots})
            total = []
            for tag in tags:
                total.extend(_toggle_fp_lots_for_tag(cardinal_ref, tag, True))
            if total and tg_logs:
                tg_logs.lots_auto_enabled("all", total)
            send(chat_id, f"🟢 Включено лотов: {len(total)}" if total else "ℹ️ Нечего включать")
        threading.Thread(target=_do, daemon=True).start()

    def lot_add(c):
        answer(c)
        _ask(c.message.chat.id, c.from_user.id,
             States.LOT_ID,
             "Введите <b>ID лота</b> или ссылку на лот:",
             _back_kb(CBT.LOTS))

    def _h_lot_id(m):
        raw = (m.text or "").strip()
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        lot_id = _extract_lot_id(raw)
        if not lot_id:
            send(m.chat.id,
                 "❌ Не удалось распознать ID лота. Введите число или ссылку funpay.com/lots/offer?id=...",
                 _back_kb(CBT.LOTS))
            return
        if SETTINGS.has_lot(lot_id):
            send(m.chat.id, f"❌ Лот <code>{lot_id}</code> уже добавлен", _back_kb(CBT.LOTS))
            return
        _temp_storage.setdefault(m.from_user.id, {})["lot_id"] = lot_id
        tags = AccountRepo.all_tags()
        if not tags:
            send(m.chat.id, "❌ Сначала добавьте аккаунты!", _main_kb())
            return
        kb = K()
        for tag in tags:
            kb.add(B(tag, None, f"{CBT.LOT_TAG}:{tag}"))
        kb.add(B("⬅️ Назад", None, CBT.LOTS))
        send(m.chat.id, "Выберите <b>тег</b>:", kb)

    def lot_tag(c):
        tag = _ntag(_p(c))
        _temp_storage.setdefault(c.from_user.id, {})["lot_tag"] = tag
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.LOT_HRS}:{h}"))
        kb.add(B("⬅️", None, CBT.LOTS))
        edit(c.message, f"Тег: <code>{tag}</code>\n\nВыберите <b>период</b>:", kb)

    def lot_hours(c):
        h = _pid(c)
        d = _temp_storage.get(c.from_user.id, {})
        lid = d.get("lot_id")
        tag = d.get("lot_tag", "default")
        if lid:
            SETTINGS.set_lot(str(lid), tag, h)
            _invalidate_lots_cache()
        edit(c.message, f"✅ Лот {lid} → <code>{tag}</code>, <code>{_period_label(h)}</code>", _main_kb())

    def open_reviews(c):
        rules = SETTINGS.get_review_rules()
        kb = K(row_width=1)
        for r in rules:
            bl = _period_label(int(r.bonus_hours)) if r.bonus_hours == int(r.bonus_hours) else f"{r.bonus_hours}ч"
            kb.add(B(f"🎁 {_period_label(r.rent_hours)} → +{bl} ❌", None, f"{CBT.REV_DEL}:{r.rent_hours}"))
        kb.add(B("➕ Добавить", None, CBT.REV_ADD))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        txt = "<b>⭐️ Бонусы за отзывы</b>\n\n"
        if rules:
            txt += "".join(f"∟ от <code>{_period_label(r.rent_hours)}</code> → <code>+{r.bonus_hours}ч</code>\n"
                           for r in rules)
            txt += "\nНажмите для удаления."
        else:
            txt += "Правил нет."
        edit(c.message, txt, kb)

    def rev_add(c):
        answer(c)
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.REV_HRS}:{h}"))
        kb.add(B("⬅️", None, CBT.REVS))
        edit(c.message, "Мин. <b>период аренды</b>:", kb)

    def rev_hours(c):
        h = _pid(c)
        _temp_storage.setdefault(c.from_user.id, {})["rev_rh"] = h
        kb = K(row_width=3)
        for bh in [1, 2, 3, 6, 12, 24]:
            kb.add(B(_period_label(bh), None, f"{CBT.REV_BON}:{bh}"))
        kb.add(B("⬅️", None, CBT.REVS))
        edit(c.message, f"Аренда от: <code>{_period_label(h)}</code>\n\n<b>Бонус</b>:", kb)

    def rev_bonus(c):
        bh = _pid(c)
        rh = _temp_storage.get(c.from_user.id, {}).get("rev_rh", 3)
        SETTINGS.add_review_rule(rh, float(bh))
        answer(c, f"✅ {_period_label(rh)} → +{_period_label(bh)}")
        open_reviews(c)

    def rev_del(c):
        SETTINGS.del_review_rule(_pid(c))
        open_reviews(c)

    def open_notifs(c):
        kb = K(row_width=1)
        for attr, label in [("notification_order_completed", "Выдача"),
                            ("notification_error", "Ошибки"),
                            ("notification_refund", "Возвраты")]:
            kb.add(B(f"{_is_on(getattr(SETTINGS, attr))} {label}", None, f"{CBT.TOGGLE}:{attr}"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, "<b>🔔 Уведомления</b>", kb)

    def open_msgs(c):
        kb = K(row_width=1)
        for key, desc in MessagesConfig.DESCRIPTIONS.items():
            kb.add(B(desc, None, f"{CBT.MSG_EDIT}:{key}"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, "<b>💬 Тексты сообщений</b>", kb)

    def msg_edit(c):
        key = _p(c)
        _temp_storage.setdefault(c.from_user.id, {})["edit_key"] = key
        answer(c)
        cur = getattr(SETTINGS.messages, key, "")
        desc = MessagesConfig.DESCRIPTIONS.get(key, "")
        txt = (
            f"<b>{desc}</b>\n\n"
            f"Текущий:\n<code>{cur}</code>\n\n"
            f"Переменные: <code>$login $password $rent_period "
            f"$code $end_time $hours $remaining $id $link $stock_list</code>\n\n"
            f"Введите новый текст:"
        )
        _ask(c.message.chat.id, c.from_user.id, States.MSG_EDIT, txt, _back_kb(CBT.MSGS))

    def _h_msg_edit(m):
        key = _temp_storage.get(m.from_user.id, {}).get("edit_key")
        _cleanup_dialog(m.chat.id, m.from_user.id, m.message_id)
        if key:
            SETTINGS.set_message(key, m.text.strip())
            send(m.chat.id, "✅ Сохранено!", _main_kb())
        else:
            send(m.chat.id, "❌ Данные утеряны, начните заново", _main_kb())

    def open_stats(c):
        kb = K(row_width=1)
        kb.add(B("📈 Полная статистика", None, CBT.FULL_STATS))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, _stats_text(), kb)

    def open_full_stats(c):
        now = time.time()
        finished = [o for o in ORDERS.values() if o.status == RentStatus.FINISHED]
        def make_block(name, from_ts):
            threshold = _fmt(MOSCOW_TZ.localize(datetime.fromtimestamp(from_ts)))
            arr = [o for o in finished if o.created_at >= threshold]
            buyers = defaultdict(lambda: {"cnt": 0, "hrs": 0.0})
            accs = defaultdict(lambda: {"cnt": 0, "hrs": 0.0})
            for o in arr:
                buyers[o.buyer]["cnt"] += 1
                buyers[o.buyer]["hrs"] += o.hours
                label = o.acc_login or f"#{o.acc_id}"
                accs[label]["cnt"] += 1
                accs[label]["hrs"] += o.hours
            def fmt_top(dct):
                top = sorted(dct.items(), key=lambda x: x[1]["hrs"], reverse=True)[:5]
                return "\n".join(f"  ∟ {k}: {v['cnt']} | {v['hrs']:.0f}ч" for k, v in top) or "  ∟ Нет данных"
            cnt = len(arr)
            hrs = sum(o.hours for o in arr)
            return (f"— <b>{name}</b>\nВсего: {cnt} аренд | {hrs:.0f} ч\n"
                    f"Покупатели:\n{fmt_top(buyers)}\nАккаунты:\n{fmt_top(accs)}\n")
        all_cnt = len(finished)
        all_hrs = sum(o.hours for o in finished)
        txt = f"📈 <b>Полная статистика</b>\n{all_cnt} аренд | {all_hrs:.0f} ч\n\n"
        txt += "\n".join([make_block("Сегодня", now - 86400),
                          make_block("Неделя", now - 604800),
                          make_block("Месяц", now - 2592000)])
        edit(c.message, txt, _back_kb(CBT.STATS))

    def open_history(c):
        page = _pid(c)
        all_orders = sorted(
            [o for o in ORDERS.values()
             if o.status in (RentStatus.FINISHED, RentStatus.REFUND,
                             RentStatus.ERROR, RentStatus.ACTIVE, RentStatus.BUSY)],
            key=lambda x: x.created_at, reverse=True)
        total = len(all_orders)
        per = 10
        pages = max(1, (total + per - 1) // per)
        page = min(max(1, page), pages)
        sl = all_orders[(page - 1) * per:page * per]
        kb = K(row_width=1)
        for o in sl:
            icons = {"FINISHED": "✅", "REFUND": "💰", "ERROR": "❌", "ACTIVE": "👤", "BUSY": "⏳"}
            icon = icons.get(o.status, "❓")
            ext = " 🔄" if o.is_extension else ""
            acc_name = o.acc_login or f"#{o.acc_id}"
            kb.add(B(f"{icon} {o.buyer} | {acc_name} | {_period_label(int(o.hours))}{ext}",
                     None, f"{CBT.HIST_DETAIL}:{o.id}"))
        if pages > 1:
            nav = []
            if page > 1:
                nav.append(B("⬅️", None, f"{CBT.HIST}:{page - 1}"))
            nav.append(B(f"{page}/{pages}", None, _CBT.EMPTY))
            if page < pages:
                nav.append(B("➡️", None, f"{CBT.HIST}:{page + 1}"))
            kb.row(*nav)
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, f"<b>📜 История ({total})</b>", kb)

    def open_history_detail(c):
        oid = _p(c)
        txt, o = _order_detail_text(oid)
        edit(c.message, txt, _back_kb(f"{CBT.HIST}:1"))

    def get_files_confirm(c):
        kb = K(row_width=2)
        kb.add(B("✅ Да, отправить", None, CBT.FILES_CONFIRM),
               B("❌ Отмена", None, CBT.MAIN))
        edit(c.message,
             "⚠️ <b>Файлы содержат пароли и секреты Steam!</b>\n\nОтправить в чат?", kb)
        answer(c)

    def get_files(c):
        answer(c)
        for f in ("settings.json", "accounts.json", "orders.json"):
            p = _get_path(f)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        bot.send_document(c.message.chat.id, fh)
                except Exception:
                    pass
    tg.cbq_handler(open_main, lambda c: c.data == CBT.MAIN or c.data.startswith(CBT.SP))
    tg.cbq_handler(open_acc_menu, lambda c: c.data == CBT.ACC_MENU)
    tg.cbq_handler(start_add, lambda c: c.data == CBT.ACC_ADD)
    tg.cbq_handler(open_lots, lambda c: c.data == CBT.LOTS)
    tg.cbq_handler(lot_add, lambda c: c.data == CBT.LOT_ADD)
    tg.cbq_handler(lots_disable_all, lambda c: c.data == CBT.LOTS_DISABLE_ALL)
    tg.cbq_handler(lots_enable_all, lambda c: c.data == CBT.LOTS_ENABLE_ALL)
    tg.cbq_handler(open_reviews, lambda c: c.data == CBT.REVS)
    tg.cbq_handler(rev_add, lambda c: c.data == CBT.REV_ADD)
    tg.cbq_handler(open_notifs, lambda c: c.data == CBT.NOTIFS)
    tg.cbq_handler(open_msgs, lambda c: c.data == CBT.MSGS)
    tg.cbq_handler(open_stats, lambda c: c.data == CBT.STATS)
    tg.cbq_handler(open_full_stats, lambda c: c.data == CBT.FULL_STATS)
    tg.cbq_handler(hrs_done, lambda c: c.data == CBT.HRS_DONE)
    for pfx, handler in [
        (CBT.ACC_LIST, open_acc_list), (CBT.ACC_DETAIL, open_acc_detail),
        (CBT.ACC_CODE, acc_code), (CBT.ACC_STOP, acc_stop),
        (CBT.ACC_CHPWD, acc_chpwd), (CBT.ACC_EXTEND_DO, acc_extend_do),
        (CBT.ACC_RESET, acc_reset), (CBT.ACC_EDIT_HOURS, acc_edit_hours),
        (CBT.ACC_TOGGLE_HOUR, acc_toggle_hour), (CBT.ACC_SAVE_HOURS, acc_save_hours),
        (CBT.ACC_MANUAL, acc_manual_start), (CBT.ACC_MANUAL_HOURS, handle_manual_hours),
        (CBT.ACC_DEL_CONFIRM, acc_del_confirm), (CBT.ACC_DEL_YES, acc_del_yes),
        (CBT.ACC_DEL_NO, acc_del_no),
        (CBT.LOT_DETAIL, open_lot_detail), (CBT.LOT_EDIT, lot_edit),
        (CBT.LOT_EDIT_TAG, lot_edit_tag), (CBT.LOT_EDIT_HRS, lot_edit_hrs),
        (CBT.LOT_RENAME, lot_rename),
        (CBT.LOT_DEL_CONFIRM, lot_del_confirm), (CBT.LOT_DEL_YES, lot_del_yes),
        (CBT.LOT_DEL_NO, lot_del_no),
        (CBT.LOT_TOGGLE_FP, lot_toggle_fp),
        (CBT.LOT_TAG, lot_tag), (CBT.LOT_HRS, lot_hours),
        (CBT.REV_HRS, rev_hours), (CBT.REV_BON, rev_bonus),
        (CBT.REV_DEL, rev_del), (CBT.MSG_EDIT, msg_edit),
        (CBT.TOGGLE, toggle_setting), (CBT.HIST, open_history),
        (CBT.HIST_DETAIL, open_history_detail),
        (CBT.HRS_TGL, hrs_toggle), (CBT.FILES, get_files_confirm),
        (CBT.FILES_CONFIRM, get_files),
        (CBT.ACC_SET_PWD, acc_set_pwd),
        (CBT.ACC_EDIT_MAFILE, acc_edit_mafile),
    ]:
        tg.cbq_handler(handler, lambda c, p=pfx: c.data.startswith(f"{p}:"))
    tg.cbq_handler(acc_extend_menu,
                   lambda c: c.data.startswith(f"{CBT.ACC_EXTEND}:") and c.data.count(":") == 1)
    for state, handler in [
        (States.LOGIN, _h_login), (States.PASS, _h_pass),
        (States.TAG, _h_tag), (States.MAN_BUYER, _h_manual_buyer),
        (States.LOT_ID, _h_lot_id), (States.MSG_EDIT, _h_msg_edit),
        (States.SET_PWD, _h_set_pwd), (States.LOT_RENAME, _h_lot_rename),
    ]:
        tg.msg_handler(handler, func=lambda m, s=state: tg.check_state(m.chat.id, m.from_user.id, s))
    tg.msg_handler(_h_mafile, func=lambda m: tg.check_state(m.chat.id, m.from_user.id, States.MAFILE))
    tg.msg_handler(_h_mafile_edit, func=lambda m: tg.check_state(m.chat.id, m.from_user.id, States.EDIT_MAFILE))
    try:
        tg.file_handler(States.MAFILE, _h_mafile)
        tg.file_handler(States.EDIT_MAFILE, _h_mafile_edit)
    except Exception:
        pass
    tg.msg_handler(open_main_cmd, commands=['auto_steam_rent'])
    card.add_telegram_commands(UUID, [
        ("auto_steam_rent", "открыть настройки авто аренды аккаунтов", True),
    ])
    threading.Thread(target=rental_check_loop, args=(card,), daemon=True).start()

def cleanup(card: Cardinal):
    _stop_event.set()

BIND_TO_PRE_INIT = [init]
BIND_TO_NEW_ORDER = [process_new_order]
BIND_TO_NEW_MESSAGE = [process_message]
BIND_TO_ORDER_STATUS_CHANGED = [process_order_status_changed]
BIND_TO_DELETE = [cleanup]