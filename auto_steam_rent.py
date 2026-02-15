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
from datetime import datetime, timedelta
from hashlib import sha1
from typing import Any, ClassVar, Dict, List, Optional, Set
from collections import defaultdict

for pkg, imp in [("aiohttp", "aiohttp"), ("pytz", "pytz"), ("pydantic", "pydantic"),
                 ("rsa", "rsa"), ("requests", "requests"), ("pysteamauth", "pysteamauth"),
                 ("steamlib", "steamlib"), ("lxml", "lxml"), ("yarl", "yarl")]:
    try:
        importlib.import_module(imp)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from playwright.async_api import async_playwright
        PLAYWRIGHT_AVAILABLE = True
    except Exception:
        pass

import aiohttp
from aiohttp import CookieJar
from yarl import URL as YarlURL
import rsa
from pytz import timezone
from pydantic import BaseModel, Field
from pysteamauth.auth import Steam as _BaseSteam

from cardinal import Cardinal
from FunPayAPI.common.enums import OrderStatuses, MessageTypes
from FunPayAPI.updater.events import NewOrderEvent, NewMessageEvent, OrderStatusChangedEvent
from tg_bot import CBT as _CBT
from telebot.types import InlineKeyboardMarkup as K, InlineKeyboardButton as B

NAME = "Auto Steam Rent"
VERSION = "0.1.0"
CREDITS = "@kewanmov"
DESCRIPTION = "Плагин для автоматической аренды Steam аккаунтов на площадке FunPay"
UUID = "b8b118dd-be5d-4697-b50e-f5f8f2e2043e"
SETTINGS_PAGE = True
PAGE_SIZE = 8

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
MAX_CODES_PER_HOUR = 30

_CMD_CODE = frozenset(("!steamguard", "!code", "/code", "код", "code"))
_CMD_TIME = frozenset(("!time", "/time", "время", "time"))

_SC_URL = YarlURL("https://steamcommunity.com")


def _period_label(h: int) -> str:
    return _PERIOD_LABELS.get(h, f"{h}ч")


def _format_periods(hours: List[int]) -> str:
    return ", ".join(_period_label(h) for h in sorted(hours))


class RentStatus:
    __slots__ = ()
    FREE = "FREE"
    BUSY = "BUSY"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"
    FINISHED = "FINISHED"
    REFUND = "REFUND"


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


os.makedirs(os.path.dirname(_get_path("")), exist_ok=True)


def _load_json(filename: str) -> Any:
    p = _get_path(filename)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_json(filename: str, data: Any):
    with open(_get_path(filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def _set_session_cookies(session: aiohttp.ClientSession, cookies: dict, url: YarlURL = _SC_URL):
    for n, v in cookies.items():
        session.cookie_jar.update_cookies({n: v}, url)


class Storage:
    __slots__ = ('_lock', '_dirty', '_timer', 'DEBOUNCE')

    def __init__(self):
        self._lock = threading.Lock()
        self._dirty: Set[str] = set()
        self._timer: Optional[threading.Timer] = None
        self.DEBOUNCE = 2.0

    def mark_dirty(self, *names: str):
        with self._lock:
            self._dirty.update(names)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.DEBOUNCE, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self):
        with self._lock:
            dirty = self._dirty.copy()
            self._dirty.clear()
        for name in dirty:
            try:
                if name == "accounts":
                    _save_json("accounts", [a.dict() for a in ACCOUNTS])
                elif name == "orders":
                    _save_json("orders", {k: v.dict() for k, v in ORDERS.items()})
                elif name == "settings":
                    _save_json("settings", SETTINGS.dict())
            except Exception as e:
                logger.error(f"Storage flush [{name}]: {e}")

    def flush_now(self):
        if self._timer:
            self._timer.cancel()
        self._flush()


storage = Storage()


class SteamGuard:
    _time_offset: int = 0
    _last_sync: float = 0
    SYNC_INTERVAL: int = 300
    SYMBOLS = "23456789BCDFGHJKMNPQRTVWXY"
    _SYM_LEN = len(SYMBOLS)
    _secret_cache: Dict[str, bytes] = {}

    @classmethod
    def _decode_secret(cls, secret: str) -> bytes:
        cached = cls._secret_cache.get(secret)
        if cached is not None:
            return cached
        padded = secret + '=' * ((4 - len(secret) % 4) % 4)
        decoded = base64.b64decode(padded)
        cls._secret_cache[secret] = decoded
        return decoded

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
    def _generate(cls, shared_secret: str) -> str:
        tw = int((int(time.time()) + cls._time_offset) / 30)
        sb = cls._decode_secret(shared_secret)
        hr = hmac.new(sb, struct.pack(">Q", tw), sha1).digest()
        o = hr[19] & 0x0F
        v = struct.unpack(">I", hr[o:o + 4])[0] & 0x7FFFFFFF
        sym, sl = cls.SYMBOLS, cls._SYM_LEN
        return ''.join(sym[v // sl ** i % sl] for i in range(5))

    @classmethod
    def _ensure_synced_sync(cls):
        if time.time() - cls._last_sync > cls.SYNC_INTERVAL:
            cls.sync_time_sync()

    @classmethod
    async def _ensure_synced_async(cls):
        if time.time() - cls._last_sync > cls.SYNC_INTERVAL:
            await cls.sync_time_async()

    @classmethod
    def code_sync(cls, shared_secret: str) -> str:
        if not shared_secret:
            return "NO_SECRET"
        cls._ensure_synced_sync()
        try:
            return cls._generate(shared_secret)
        except Exception:
            return "ERROR"

    @classmethod
    async def code_async(cls, shared_secret: str) -> str:
        if not shared_secret:
            return "NO_SECRET"
        await cls._ensure_synced_async()
        try:
            return cls._generate(shared_secret)
        except Exception:
            return "ERROR"


def _conf_key(identity_secret: str, timestamp: int, tag: str) -> str:
    sb = SteamGuard._decode_secret(identity_secret)
    return base64.b64encode(
        hmac.new(sb, struct.pack(">Q", timestamp) + tag.encode(), sha1).digest()
    ).decode()


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

    async def json_request(self, url, method="GET", **kw):
        return json.loads(await self.request(url, method, **kw))

    async def raw_request(self, url, method="GET", **kw):
        from urllib3.util import parse_url
        return await self._requests.request(
            url=url, method=method,
            cookies=await self.cookies(parse_url(url).host), **kw)


class SteamPasswordChanger:
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    _HELP = "https://help.steampowered.com/en/wizard"
    _AJAX_HEADERS = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://help.steampowered.com",
        "X-Requested-With": "XMLHttpRequest",
    }
    _COOKIE_DOMAINS = ("help.steampowered.com", "store.steampowered.com", "steamcommunity.com")

    def __init__(self, mafile: dict, current_password: str):
        self.mafile = mafile
        self.current_password = current_password
        self.login = mafile.get("account_name", "")
        self.shared_secret = mafile.get("shared_secret", "")
        self.identity_secret = mafile.get("identity_secret", "")
        self.device_id = mafile.get("device_id", "")
        self.steamid = int(mafile.get("Session", {}).get("SteamID", 0))
        self._steam: Optional[CustomSteam] = None

    def _headers(self) -> dict:
        return {**self._AJAX_HEADERS, "User-Agent": self.UA}

    async def _steam_req(self, endpoint: str, method: str = "POST",
                         check_error: bool = True, **kwargs) -> dict:
        sid = await self._steam.sessionid("help.steampowered.com")
        payload_key = "data" if method == "POST" else "params"
        kwargs.setdefault(payload_key, {})["sessionid"] = sid
        r = await self._steam.json_request(
            method=method, url=f"{self._HELP}/{endpoint}",
            headers=self._headers(), **kwargs)
        if check_error and r.get("errorMsg"):
            raise Exception(f"{endpoint}: {r['errorMsg']}")
        return r

    async def change_password(self) -> str:
        if not PLAYWRIGHT_AVAILABLE:
            raise Exception("Playwright не установлен!")
        new_password = _gen_password(20)
        self._steam = CustomSteam(
            self.login, self.current_password, self.shared_secret,
            self.identity_secret, self.device_id, self.steamid)
        await self._do_login()
        params = await self._get_change_params()
        await self._pw_trigger_confirmation(params)
        await asyncio.sleep(5)
        if not await self._confirm_recovery(params):
            raise Exception("Mobile confirmation failed")
        await self._poll_recovery(params)

        s, reset, lost, issueid, account = (
            params.get("s"), params.get("reset"), params.get("lost", 0),
            params.get("issueid"), params.get("account"))

        await self._steam_req("AjaxVerifyAccountRecoveryCode", method="GET",
                              params={"code": "", "s": s, "reset": reset,
                                      "lost": lost, "method": 8,
                                      "issueid": issueid, "wizard_ajax": 1, "gamepad": 0})
        await self._steam_req("AjaxAccountRecoveryGetNextStep",
                              data={"wizard_ajax": 1, "s": s, "account": account,
                                    "reset": reset, "issueid": issueid, "lost": 2})
        key = await self._get_rsa_key()
        enc_old = self._encrypt(self.current_password, key["publickey_mod"], key["publickey_exp"])
        await self._steam_req("AjaxAccountRecoveryVerifyPassword/",
                              data={"s": s, "lost": 2, "reset": 1,
                                    "password": enc_old, "rsatimestamp": key["timestamp"]})
        key = await self._get_rsa_key()
        await self._steam_req("AjaxCheckPasswordAvailable/",
                              data={"wizard_ajax": 1, "password": new_password})
        enc_new = self._encrypt(new_password, key["publickey_mod"], key["publickey_exp"])
        await self._steam_req("AjaxAccountRecoveryChangePassword/",
                              data={"wizard_ajax": 1, "s": s, "account": account,
                                    "password": enc_new, "rsatimestamp": key["timestamp"]})
        return new_password

    async def _do_login(self):
        for attempt in range(3):
            try:
                await SteamGuard.sync_time_async()
                await asyncio.sleep(1)
                await self._steam.login_to_steam()
                return
            except Exception as e:
                err = str(e)
                if "TwoFactorCodeMismatch" in err:
                    await asyncio.sleep(5)
                elif "RateLimitExceeded" in err:
                    await asyncio.sleep(30 * (attempt + 1))
                else:
                    raise
        raise Exception("Steam login failed after 3 attempts")

    async def _get_change_params(self) -> dict:
        resp = await self._steam.raw_request(
            method="GET",
            url="https://help.steampowered.com/wizard/HelpChangePassword?redir=store/account/",
            headers={"Accept": "text/html,application/xhtml+xml",
                     "Referer": "https://store.steampowered.com/",
                     "User-Agent": self.UA},
            allow_redirects=True)
        if resp.history:
            try:
                q = dict(YarlURL(resp.real_url).query)
                result = {}
                for k, v in q.items():
                    try:
                        result[k] = int(v)
                    except (ValueError, TypeError):
                        result[k] = v
                for dk in ("lost", "s", "account", "reset", "issueid"):
                    result.setdefault(dk, 0)
                return result
            except Exception:
                pass
        html = await resp.text()
        try:
            from lxml.html import document_fromstring
            errors = document_fromstring(html).cssselect("#error_description")
            if errors:
                raise Exception(f"Steam: {errors[0].text}")
        except ImportError:
            pass
        raise Exception("Failed to get password change params")

    async def _get_cookies_dict(self) -> dict:
        cookies = {}
        for domain in self._COOKIE_DOMAINS:
            try:
                for n, v in (await self._steam.cookies(domain)).items():
                    cookies[n] = v
            except Exception:
                pass
        return cookies

    async def _pw_trigger_confirmation(self, params: dict):
        cookies_for_pw = []
        for domain in self._COOKIE_DOMAINS:
            try:
                for n, v in (await self._steam.cookies(domain)).items():
                    cookies_for_pw.append({"name": n, "value": v, "domain": f".{domain}", "path": "/"})
            except Exception:
                pass
        url = (f"https://help.steampowered.com/en/wizard/HelpWithLoginInfoEnterCode"
               f"?s={params.get('s', 0)}&account={params.get('account', 0)}"
               f"&reset={params.get('reset', 0)}&lost={params.get('lost', 0)}"
               f"&issueid={params.get('issueid', 0)}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-setuid-sandbox",
                                     "--disable-dev-shm-usage", "--disable-gpu"])
            ctx = await browser.new_context(user_agent=self.UA, locale="en-US",
                                            viewport={"width": 1280, "height": 720})
            if cookies_for_pw:
                await ctx.add_cookies(cookies_for_pw)
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(5)
            except Exception:
                pass
            finally:
                await browser.close()

    async def _get_confirmations_with_session(self, session: aiohttp.ClientSession) -> list:
        await SteamGuard.sync_time_async()
        ts = int(time.time()) + SteamGuard._time_offset
        ck = _conf_key(self.identity_secret, ts, "getlist")
        p = {
            "p": self.device_id, "a": str(self.steamid),
            "k": ck, "t": str(ts), "m": "react", "tag": "getlist"
        }
        try:
            async with session.get("https://steamcommunity.com/mobileconf/getlist", params=p) as resp:
                data = await resp.json()
                return data.get("conf", []) if data.get("success") else []
        except Exception:
            return []

    async def _accept_confirmation_with_session(self, session: aiohttp.ClientSession, conf: dict) -> bool:
        await SteamGuard.sync_time_async()
        ts = int(time.time()) + SteamGuard._time_offset
        ak = _conf_key(self.identity_secret, ts, "allow")
        ap = {
            "p": self.device_id, "a": str(self.steamid),
            "k": ak, "t": str(ts), "m": "react", "tag": "allow",
            "op": "allow", "cid": str(conf["id"]), "ck": conf["nonce"]
        }
        try:
            async with session.get("https://steamcommunity.com/mobileconf/ajaxop", params=ap) as resp:
                result = await resp.json()
                return result.get("success", False)
        except Exception:
            return False

    def _make_session_with_cookies(self, cookies: dict) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession()
        _set_session_cookies(session, cookies, _SC_URL)
        return session

    async def _confirm_recovery(self, params: dict) -> bool:
        cookies = await self._get_cookies_dict()
        async with self._make_session_with_cookies(cookies) as session:
            try:
                from steamlib.api.trade import SteamTrade
                from steamlib.api.trade.exceptions import NotFoundMobileConfirmationError
                st = SteamTrade(self._steam)
                for _ in range(20):
                    try:
                        confs = await self._get_confirmations_with_session(session)
                        for conf in confs:
                            if await self._accept_confirmation_with_session(session, conf):
                                return True
                    except Exception:
                        pass
                    try:
                        if await st.mobile_confirm_by_creator_id(params.get("s", 0)):
                            return True
                    except NotFoundMobileConfirmationError:
                        pass
                    except Exception:
                        pass
                    await asyncio.sleep(3)
            except ImportError:
                pass
            return await self._manual_confirm_with_session(session)

    async def _manual_confirm_with_session(self, session: aiohttp.ClientSession) -> bool:
        for _ in range(25):
            try:
                confs = await self._get_confirmations_with_session(session)
                for conf in confs:
                    if conf.get("type", 0) in (1, 2, 3, 6):
                        if await self._accept_confirmation_with_session(session, conf):
                            return True
            except Exception:
                pass
            await asyncio.sleep(3)
        return False

    async def _poll_recovery(self, params: dict) -> bool:
        for _ in range(15):
            try:
                r = await self._steam_req(
                    "AjaxPollAccountRecoveryConfirmation", check_error=False,
                    data={"wizard_ajax": 1, "s": params.get("s"), "reset": params.get("reset"),
                          "lost": params.get("lost", 0), "method": 8,
                          "issueid": params.get("issueid"), "gamepad": 0})
                if r.get("success") or r.get("continue"):
                    return True
                if r.get("errorMsg"):
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        return False

    async def _get_rsa_key(self) -> dict:
        sid = await self._steam.sessionid("help.steampowered.com")
        return await self._steam.json_request(
            method="POST", url="https://help.steampowered.com/en/login/getrsakey/",
            data={"sessionid": sid, "username": self.login}, headers=self._headers())

    @staticmethod
    def _encrypt(password: str, mod: str, exp: str) -> str:
        pk = rsa.PublicKey(n=int(mod, 16), e=int(exp, 16))
        return base64.b64encode(rsa.encrypt(password.encode("ascii"), pk)).decode()


async def change_password_async(mafile: dict, current_password: str) -> str:
    return await SteamPasswordChanger(mafile, current_password).change_password()


def change_password_sync(mafile: dict, current_password: str) -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(change_password_async(mafile, current_password))
    finally:
        loop.close()


class LotConfig(BaseModel):
    tag: str
    hours: int

    class Config:
        extra = "allow"


class MessagesConfig(BaseModel):
    order_completed: str = ("✅ Данные от аккаунта:\n∟ Логин: $login\n∟ Пароль: $password\n"
                            "∟ Аренда на: $rent_period\n\n⚠️ Для входа вам понадобится Steam Guard код.\n"
                            "Напишите !code чтобы получить код")
    guard_code: str = "✅ Steam Guard код: $code\n∟ Действителен ~30 секунд\n∟ Аренда до: $end_time"
    rent_over: str = "⛔ Аренда завершена!\n∟ Пароль изменён"
    warning: str = "⚠️ Аренда заканчивается через 10 минут!"
    extended: str = "✅ Аренда продлена на +$hours ч.\n∟ Окончание: $end_time"
    bonus: str = "✅ Бонус за отзыв: +$hours ч."
    time_info: str = "✅ Осталось: $remaining\n∟ Окончание: $end_time"
    error_msg: str = "❌ Произошла ошибка! Ожидайте ответа продавца"
    no_accounts: str = "❌ Нет свободных аккаунтов! Средства будут возвращены"
    refunded: str = "✅ Средства возвращены"
    rent_expired: str = "⛔ Время аренды истекло!"
    no_order: str = "❌ Активный заказ не найден"
    no_account: str = "❌ Аккаунт не найден"
    code_error: str = "❌ Ошибка генерации кода, попробуйте через 30 сек"
    config_error: str = "❌ Ошибка конфигурации, обратитесь к продавцу"
    rent_not_started: str = "⚠️ Напишите !code чтобы начать аренду"
    DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "order_completed": "📋 Выдача данных", "guard_code": "🔑 Steam Guard код",
        "rent_over": "⛔ Конец аренды", "warning": "⚠️ Предупреждение 10 мин",
        "extended": "✅ Продление", "bonus": "🎁 Бонус за отзыв",
        "time_info": "⏱ Команда !time", "rent_expired": "⏰ Время истекло",
        "error_msg": "❌ Общая ошибка", "no_accounts": "❌ Нет аккаунтов",
        "refunded": "💰 Возврат", "no_order": "❌ Заказ не найден",
        "no_account": "❌ Аккаунт не найден", "code_error": "❌ Ошибка кода",
        "config_error": "❌ Ошибка конфигурации", "rent_not_started": "⏰ Аренда не начата",
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
    allowed_hours: List[int] = Field(default_factory=lambda: [24])
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


class RentOrder(BaseModel):
    id: str
    chat_id: Any
    buyer: str
    buyer_id: int
    acc_id: int
    hours: float
    status: str = RentStatus.BUSY
    warned: bool = False
    review_claimed: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        extra = "allow"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        storage.mark_dirty("orders")


class Settings(BaseModel):
    enabled: bool = True
    autoback_on_error: bool = True
    lots: Dict[str, Any] = Field(default_factory=dict)
    review_rules: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"rent_hours": 3, "bonus_hours": 1.0}, {"rent_hours": 6, "bonus_hours": 2.0},
        {"rent_hours": 12, "bonus_hours": 4.0}, {"rent_hours": 24, "bonus_hours": 6.0},
        {"rent_hours": 72, "bonus_hours": 12.0}, {"rent_hours": 168, "bonus_hours": 24.0},
    ])
    messages: MessagesConfig = MessagesConfig()
    notification_order_completed: bool = True
    notification_error: bool = True
    notification_refund: bool = True

    class Config:
        extra = "allow"

    def toggle(self, p: str):
        setattr(self, p, not getattr(self, p))
        storage.mark_dirty("settings")

    def set_message(self, k: str, v: str):
        setattr(self.messages, k, v)
        storage.mark_dirty("settings")

    def get_lot(self, lot_id: str) -> Optional[LotConfig]:
        raw = self.lots.get(str(lot_id))
        if raw is None:
            return None
        if isinstance(raw, str):
            return LotConfig(tag=_ntag(raw), hours=24)
        if isinstance(raw, dict):
            return LotConfig(**raw)
        return None

    def set_lot(self, lot_id: str, tag: str, hours: int):
        self.lots[str(lot_id)] = {"tag": _ntag(tag), "hours": hours}
        storage.mark_dirty("settings")

    def del_lot(self, lot_id: str):
        self.lots.pop(str(lot_id), None)
        storage.mark_dirty("settings")

    def get_review_rules(self) -> List[ReviewRule]:
        return sorted([ReviewRule(**r) for r in self.review_rules if isinstance(r, dict)],
                      key=lambda x: x.rent_hours)

    def add_review_rule(self, rent_hours: int, bonus_hours: float):
        self.review_rules = [r for r in self.review_rules
                             if not (isinstance(r, dict) and r.get("rent_hours") == rent_hours)]
        self.review_rules.append({"rent_hours": rent_hours, "bonus_hours": bonus_hours})
        storage.mark_dirty("settings")

    def del_review_rule(self, rent_hours: int):
        self.review_rules = [r for r in self.review_rules
                             if not (isinstance(r, dict) and r.get("rent_hours") == rent_hours)]
        storage.mark_dirty("settings")

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
_processed_orders: Set[str] = set()


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
    if changed:
        storage.mark_dirty("settings")
    d = _load_json("accounts")
    if isinstance(d, list):
        for a in d:
            if "rent_hours" in a and "allowed_hours" not in a:
                a["allowed_hours"] = [a["rent_hours"]]
        ACCOUNTS = [AccountModel(**a) for a in d]
    else:
        ACCOUNTS = []
    d = _load_json("orders")
    ORDERS = {k: RentOrder(**v) for k, v in d.items()} if isinstance(d, dict) else {}
    _processed_orders.update(ORDERS.keys())


_load_all()


class AccountRepo:
    _lock = threading.Lock()
    _by_id: Dict[int, AccountModel] = {}
    _by_order: Dict[str, AccountModel] = {}
    _free_by_tag: Dict[str, List[AccountModel]] = defaultdict(list)
    _order_by_chat: Dict[str, RentOrder] = {}

    @classmethod
    def rebuild(cls):
        cls._by_id = {a.id: a for a in ACCOUNTS}
        cls._by_order = {a.current_order: a for a in ACCOUNTS if a.current_order}
        cls._rebuild_free_index()
        cls._rebuild_chat_index()

    @classmethod
    def _rebuild_free_index(cls):
        idx: Dict[str, List[AccountModel]] = defaultdict(list)
        for a in ACCOUNTS:
            if a.status == RentStatus.FREE:
                idx[_ntag(a.tag)].append(a)
        cls._free_by_tag = idx

    @classmethod
    def _rebuild_chat_index(cls):
        idx: Dict[str, RentOrder] = {}
        for o in ORDERS.values():
            if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                continue
            if o.chat_id is not None:
                idx[str(o.chat_id)] = o
        cls._order_by_chat = idx

    @classmethod
    def _save(cls):
        cls.rebuild()
        storage.mark_dirty("accounts")

    @classmethod
    def get(cls, acc_id: int) -> Optional[AccountModel]:
        return cls._by_id.get(acc_id)

    @classmethod
    def by_order(cls, order_id: str) -> Optional[AccountModel]:
        return cls._by_order.get(order_id)

    @classmethod
    def get_free(cls, tag: str, hours: int = None) -> Optional[AccountModel]:
        tag = _ntag(tag)
        for a in cls._free_by_tag.get(tag, ()):
            if hours is None or hours in a.allowed_hours:
                return a
        return None

    @classmethod
    def find_order_by_chat(cls, chat_id, author_id=None, author_name=None) -> Optional[RentOrder]:
        key = str(chat_id)

        order = cls._order_by_chat.get(key)
        if order:
            return order

        for chat_key, o in cls._order_by_chat.items():
            if chat_key.startswith("users-") and key in chat_key.split("-")[1:]:
                return o

        if author_id and author_id > 0:
            for o in ORDERS.values():
                if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                    continue
                if o.buyer_id == author_id:
                    return o

        if author_name:
            author_lower = author_name.strip().lower()
            for o in ORDERS.values():
                if o.status in (RentStatus.FINISHED, RentStatus.REFUND):
                    continue
                if o.buyer and o.buyer.strip().lower() == author_lower:
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

    @classmethod
    def add(cls, login, password, mafile, tag, allowed_hours=None):
        if allowed_hours is None:
            allowed_hours = [24]
        tag = _ntag(tag)
        with cls._lock:
            if any(a.login.lower() == login.lower() for a in ACCOUNTS):
                return False, "Аккаунт уже существует"
            nid = max((a.id for a in ACCOUNTS), default=0) + 1
            ACCOUNTS.append(AccountModel(
                id=nid, login=login, password=password, mafile=mafile,
                tag=tag, allowed_hours=sorted(allowed_hours), rent_hours=allowed_hours[0]))
            cls._save()
        return True, (f"Аккаунт {login} добавлен (ID: {nid}, тег: {tag}, "
                      f"периоды: {_format_periods(allowed_hours)})")

    @classmethod
    def delete(cls, acc_id: int) -> bool:
        with cls._lock:
            for i, a in enumerate(ACCOUNTS):
                if a.id == acc_id:
                    del ACCOUNTS[i]
                    cls._save()
                    return True
        return False

    @classmethod
    def assign(cls, acc_id, order_id, buyer, buyer_id, chat_id, hours):
        with cls._lock:
            acc = cls.get(acc_id)
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
            cls._save()

    @classmethod
    def start_rent(cls, order_id) -> Optional[AccountModel]:
        with cls._lock:
            acc = cls.by_order(order_id)
            if acc:
                acc.status = RentStatus.ACTIVE
                acc.rental_end = _fmt(_now() + timedelta(hours=acc.rent_hours))
                cls._save()
            return acc

    @classmethod
    def extend_rent(cls, acc_id: int, hours: float) -> Optional[str]:
        with cls._lock:
            acc = cls.get(acc_id)
            if acc and acc.rental_end:
                acc.rental_end = _fmt(_parse(acc.rental_end) + timedelta(hours=hours))
                cls._save()
                return acc.rental_end
        return None

    @classmethod
    def release(cls, acc_id: int, new_password: str = None, error: bool = False):
        with cls._lock:
            acc = cls.get(acc_id)
            if not acc:
                return
            acc.status = RentStatus.ERROR if error else RentStatus.FREE
            acc.current_order = acc.owner = acc.owner_id = None
            acc.owner_chat_id = acc.rental_start = acc.rental_end = None
            acc.access_count = 0
            if new_password:
                acc.password = new_password
            cls._save()

    @classmethod
    def reset_to_free(cls, acc_id: int):
        with cls._lock:
            acc = cls.get(acc_id)
            if not acc:
                return
            acc.status = RentStatus.FREE
            acc.current_order = acc.owner = acc.owner_id = None
            acc.owner_chat_id = acc.rental_start = acc.rental_end = None
            acc.access_count = 0
            cls._save()

    @classmethod
    def manual_assign(cls, acc_id: int, buyer: str, hours: int) -> Optional[AccountModel]:
        with cls._lock:
            acc = cls.get(acc_id)
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
                                    acc_id=acc.id, hours=float(hours), status=RentStatus.ACTIVE)
            cls._save()
            storage.mark_dirty("orders")
        return acc

    @classmethod
    def update_allowed_hours(cls, acc_id: int, allowed_hours: List[int]) -> bool:
        with cls._lock:
            acc = cls.get(acc_id)
            if not acc:
                return False
            acc.allowed_hours = sorted(allowed_hours)
            if acc.status == RentStatus.FREE:
                acc.rent_hours = allowed_hours[0] if allowed_hours else 24
            cls._save()
        return True

    @classmethod
    def get_stats(cls) -> dict:
        r = {s: 0 for s in (RentStatus.FREE, RentStatus.ACTIVE, RentStatus.BUSY, RentStatus.ERROR)}
        for a in ACCOUNTS:
            if a.status in r:
                r[a.status] += 1
        r["total"] = len(ACCOUNTS)
        return r

    @classmethod
    def all_tags(cls) -> List[str]:
        return list({_ntag(a.tag) for a in ACCOUNTS})


AccountRepo.rebuild()


class TgLogs:
    __slots__ = ('c', 'bot')

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

    def lot_debug(self, order_id, description, lot_id, found):
        self._send(f"🔍 Поиск лота для #{order_id[:8]}...\n"
                   f"∟ Описание: <code>{description[:80]}</code>\n"
                   f"∟ Найден lot_id: <code>{lot_id}</code>\n"
                   f"∟ В конфиге: {'✅' if found else '❌'}")


def _tmpl(template: str, **kw) -> str:
    r = template
    for k, v in kw.items():
        r = r.replace(f"${k}", str(v))
    return r


def _send_fp(c, chat_id, text):
    try:
        c.send_message(chat_id, text)
    except Exception:
        pass


def _do_refund(c, order_id) -> bool:
    try:
        c.account.refund(order_id)
        return True
    except Exception as e:
        s = str(e).lower()
        return "уже" in s or "already" in s


def _build_lot_id_index(c) -> Dict[str, int]:
    index = {}
    try:
        lots = c.account.get_user(c.account.id).get_lots()
        for lot in lots:
            lid = lot.id
            if lid is None:
                continue
            lid = int(lid) if isinstance(lid, str) and str(lid).isdigit() else lid
            title = (lot.description or lot.title or "").strip()
            if title:
                index[title] = lid
    except Exception as e:
        logger.error(f"Failed to build lot index: {e}")
    return index


def _resolve_lot_id(c, order) -> Optional[str]:
    order_id = getattr(order, 'id', '?')

    for attr in ('lot_id', 'offer_id'):
        val = getattr(order, attr, None)
        if val:
            lid = str(val)
            if SETTINGS.get_lot(lid):
                logger.info(f"Order {order_id}: direct lot_id={lid} from order.{attr}, found in config")
                return lid
            else:
                logger.debug(f"Order {order_id}: direct lot_id={lid} from order.{attr}, NOT in config")

    for attr in ('lot', 'offer', 'item'):
        obj = getattr(order, attr, None)
        if obj and hasattr(obj, 'id') and obj.id:
            lid = str(obj.id)
            if SETTINGS.get_lot(lid):
                logger.info(f"Order {order_id}: lot_id={lid} from order.{attr}.id, found in config")
                return lid

    description = ""
    for attr in ('description', 'title', 'short_description', 'lot_title'):
        val = getattr(order, attr, None)
        if val and isinstance(val, str) and val.strip():
            description = val.strip()
            break

    if not description:
        logger.warning(f"Order {order_id}: no description, cannot resolve lot")
        return None

    logger.debug(f"Order {order_id}: resolving by description='{description}'")

    try:
        title_to_id = _build_lot_id_index(c)
        if not title_to_id:
            logger.warning(f"Order {order_id}: no lots found on profile")
            return None

        matched_lid = title_to_id.get(description)
        if matched_lid is not None:
            lid = str(matched_lid)
            if SETTINGS.get_lot(lid):
                logger.info(f"Order {order_id}: exact title match -> lot_id={lid}")
                return lid
            else:
                logger.info(f"Order {order_id}: exact title match -> lot_id={lid}, but NOT in config, skipping")
                return None

        logger.info(f"Order {order_id}: no exact match for '{description[:60]}' among {len(title_to_id)} lots. "
                    f"Configured: {list(SETTINGS.lots.keys())}")
        return None

    except Exception as e:
        logger.error(f"Order {order_id}: error resolving lot: {e}")
        return None


def _recover_account(c, acc, order, reason):
    try:
        np = change_password_sync(acc.mafile, acc.password)
        AccountRepo.release(acc.id, np)
        if order:
            order.update(status=RentStatus.FINISHED)
            if reason == "TIME" and order.chat_id:
                _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.rent_over, id=order.id))
    except Exception as e:
        AccountRepo.release(acc.id, error=True)
        if tg_logs:
            tg_logs.error(f"Смена пароля: {acc.login} - {str(e)[:50]}")
        if SETTINGS.autoback_on_error and order:
            if _do_refund(c, order.id):
                order.update(status=RentStatus.REFUND)
            if tg_logs:
                tg_logs.refund(order.id, f"Ошибка: {acc.login}")


def process_new_order(c, event):
    if not SETTINGS or not SETTINGS.enabled:
        return
    order = event.order
    if not order:
        return

    order_id = getattr(order, 'id', None)
    if not order_id:
        return

    if order_id in _processed_orders:
        logger.debug(f"Order {order_id} already processed, skipping")
        return
    _processed_orders.add(order_id)

    if order_id in ORDERS:
        logger.debug(f"Order {order_id} already in ORDERS, skipping")
        return

    lot_id = _resolve_lot_id(c, order)

    description = getattr(order, 'description', None) or getattr(order, 'title', None) or ""

    if not lot_id:
        logger.info(f"Order {order_id}: lot not found or not in config, description='{description[:60]}', ignoring")
        if tg_logs:
            tg_logs.lot_debug(order_id, description, "не найден", False)
        return

    lot_cfg = SETTINGS.get_lot(lot_id)
    if not lot_cfg:
        logger.info(f"Order {order_id}: lot_id={lot_id} not in config, ignoring")
        if tg_logs:
            tg_logs.lot_debug(order_id, description, lot_id, False)
        return

    logger.info(f"Order {order_id}: lot_id={lot_id}, tag={lot_cfg.tag}, hours={lot_cfg.hours}")
    if tg_logs:
        tg_logs.lot_debug(order_id, description, lot_id, True)

    tag, hours = _ntag(lot_cfg.tag), lot_cfg.hours
    if hours <= 0:
        hours = 24

    acc = AccountRepo.get_free(tag, hours)
    if not acc:
        if SETTINGS.autoback_on_error:
            _send_fp(c, order.chat_id, SETTINGS.messages.no_accounts)
            if _do_refund(c, order.id):
                _send_fp(c, order.chat_id, SETTINGS.messages.refunded)
            if tg_logs:
                tg_logs.refund(order.id, f"Нет аккаунтов (тег: {tag}, часы: {hours})")
        return

    AccountRepo.assign(acc.id, order.id, order.buyer_username, order.buyer_id, order.chat_id, hours)
    ro = RentOrder(id=order.id, chat_id=order.chat_id, buyer=order.buyer_username,
                   buyer_id=order.buyer_id, acc_id=acc.id, hours=float(hours))
    ORDERS[order.id] = ro
    storage.mark_dirty("orders")
    AccountRepo._rebuild_chat_index()
    _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.order_completed,
                                     login=acc.login, password=acc.password, id=order.id,
                                     rent_period=_period_label(hours)))
    if tg_logs:
        tg_logs.order_completed(ro, acc.login)


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
    if not (is_code or is_time):
        return

    author_name = getattr(msg, 'author', None) or getattr(msg, 'author_username', None)
    author_id = getattr(msg, 'author_id', None) or 0

    order = AccountRepo.find_order_by_chat(msg.chat_id, author_id, author_name)
    if not order:
        _send_fp(c, msg.chat_id, SETTINGS.messages.no_order)
        return

    acc = AccountRepo.by_order(order.id) or AccountRepo.get(order.acc_id)
    if not acc:
        _send_fp(c, msg.chat_id, SETTINGS.messages.no_account)
        return

    if order.chat_id != msg.chat_id:
        order.chat_id = msg.chat_id
        storage.mark_dirty("orders")
        AccountRepo._rebuild_chat_index()
        if acc.owner_chat_id != msg.chat_id:
            acc.owner_chat_id = msg.chat_id
            storage.mark_dirty("accounts")

    if is_code:
        cd_key = str(msg.chat_id)
        now_ts = time.time()
        last = _code_cooldowns.get(cd_key)
        if last is not None and now_ts - last < CODE_COOLDOWN:
            return
        _code_cooldowns[cd_key] = now_ts

        if acc.access_count > 0 and acc.rental_start:
            hours_passed = max(0.1, (_now() - _parse(acc.rental_start)).total_seconds() / 3600)
            if acc.access_count / hours_passed > MAX_CODES_PER_HOUR:
                _send_fp(c, msg.chat_id, "⚠️ Слишком много запросов кода. Подождите.")
                return

        ss = acc.mafile.get("shared_secret", "")
        if not ss:
            _send_fp(c, msg.chat_id, SETTINGS.messages.config_error)
            return
        code = SteamGuard.code_sync(ss)
        if code in ("ERROR", "NO_SECRET"):
            _send_fp(c, msg.chat_id, SETTINGS.messages.code_error)
            return
        if order.status == RentStatus.BUSY and AccountRepo.start_rent(order.id):
            order.update(status=RentStatus.ACTIVE)
        _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.guard_code,
                                       code=code, end_time=acc.rental_end or "?"))
        acc.access_count += 1
        storage.mark_dirty("accounts")

    elif is_time:
        if not acc.rental_end:
            _send_fp(c, msg.chat_id, SETTINGS.messages.rent_not_started)
            return
        rem = (_parse(acc.rental_end) - _now()).total_seconds()
        if rem <= 0:
            _send_fp(c, msg.chat_id, SETTINGS.messages.rent_expired)
        else:
            _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.time_info,
                                           remaining=_remaining_str(acc.rental_end),
                                           end_time=acc.rental_end))


def _handle_feedback(c, message):
    from FunPayAPI.common.utils import RegularExpressions
    oids = RegularExpressions().ORDER_ID.findall(message.text or "")
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
    acc = AccountRepo.by_order(event.order.id) or AccountRepo.get(order.acc_id)
    if not acc:
        return
    if event.order.status == OrderStatuses.REFUNDED:
        _recover_account(c, acc, order, "REFUND_EXT")
    elif event.order.status == OrderStatuses.CLOSED:
        order.update(status=RentStatus.FINISHED)


def rental_check_loop(c):
    cleanup_counter = 0
    while True:
        try:
            now = _now()
            for acc in list(ACCOUNTS):
                if acc.status not in (RentStatus.ACTIVE, RentStatus.BUSY) or not acc.current_order:
                    continue
                order = ORDERS.get(acc.current_order)
                if not order:
                    AccountRepo.release(acc.id)
                    continue
                if not order.id.startswith("manual"):
                    try:
                        fo = c.account.get_order(order.id)
                        if fo.status == OrderStatuses.REFUNDED:
                            _recover_account(c, acc, order, "REFUND")
                            continue
                        if fo.status == OrderStatuses.CLOSED and order.status != RentStatus.FINISHED:
                            order.update(status=RentStatus.FINISHED)
                            continue
                    except Exception:
                        pass
                if acc.rental_end:
                    rem = (_parse(acc.rental_end) - now).total_seconds()
                    if 0 < rem < 600 and not order.warned:
                        if order.chat_id:
                            _send_fp(c, order.chat_id, SETTINGS.messages.warning)
                        order.update(warned=True)
                    if rem <= 0:
                        _recover_account(c, acc, order, "TIME")
        except Exception:
            pass

        cleanup_counter += 1
        if cleanup_counter >= 30:
            cleanup_counter = 0
            try:
                cutoff = (_now() - timedelta(days=30)).isoformat()
                old_keys = [k for k, o in ORDERS.items()
                            if o.status in (RentStatus.FINISHED, RentStatus.REFUND)
                            and o.created_at < cutoff]
                if old_keys:
                    for k in old_keys:
                        del ORDERS[k]
                        _processed_orders.discard(k)
                    storage.mark_dirty("orders")
                    AccountRepo._rebuild_chat_index()
            except Exception:
                pass
            try:
                now_ts = time.time()
                stale = [k for k, v in _code_cooldowns.items() if now_ts - v > 3600]
                for k in stale:
                    del _code_cooldowns[k]
            except Exception:
                pass

        time.sleep(60)


class CBT:
    SP = f'{_CBT.PLUGIN_SETTINGS}:{UUID}'
    MAIN = "asr_main"
    ACC_MENU = "asr_accs"
    ACC_ADD = "asr_add"
    ACC_DEL = "asr_del"
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
    LOTS = "asr_lots"
    LOT_ADD = "asr_ladd"
    LOT_TAG = "asr_ltag"
    LOT_HRS = "asr_lhrs"
    LOT_DEL = "asr_ldel"
    REVS = "asr_revs"
    REV_ADD = "asr_radd"
    REV_DEL = "asr_rdel"
    REV_HRS = "asr_rhrs"
    REV_BON = "asr_rbon"
    NOTIFS = "asr_ntf"
    MSGS = "asr_msgs"
    MSG_EDIT = "asr_medt"
    STATS = "asr_stat"
    HIST = "asr_hist"
    TOGGLE = "asr_tgl"
    FILES = "asr_files"
    DEBUG = "asr_dbg"
    HRS_TGL = "asr_htgl"
    HRS_DONE = "asr_hdone"


class TelegramUI:
    def __init__(self, card: Cardinal):
        self.card = card
        self.bot = card.telegram.bot
        self.tg = card.telegram
        self._tmp: Dict[int, dict] = {}

    def _send(self, cid, text, kb=None):
        real_id = cid.chat.id if hasattr(cid, 'chat') else cid
        return self.bot.send_message(real_id, text, reply_markup=kb, parse_mode='HTML')

    def _edit(self, m, text, kb=None):
        try:
            return self.bot.edit_message_text(text, m.chat.id, m.message_id,
                                              reply_markup=kb, parse_mode='HTML')
        except Exception as e:
            if "message is not modified" not in str(e):
                raise

    def _answer(self, cb, msg=None, alert=False):
        return self.bot.answer_callback_query(cb.id, msg, show_alert=alert)

    def _p(self, c, idx=-1) -> str:
        return c.data.split(":")[idx]

    def _pid(self, c, idx=-1) -> int:
        return int(self._p(c, idx))

    def _del_msg(self, cid, mid):
        try:
            self.bot.delete_message(cid, mid)
        except Exception:
            pass

    def _back_kb(self, cb=None):
        return K().add(B("⬅️ Назад", None, cb or CBT.MAIN))

    def _hours_kb(self, selected, toggle_cb, done_cb, back_cb):
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            check = "✅" if h in selected else "⬜"
            kb.add(B(f"{check} {_period_label(h)}", None, f"{toggle_cb}:{h}"))
        kb.add(B("✅ Готово", None, done_cb))
        kb.add(B("⬅️ Назад", None, back_cb))
        return kb

    def _get_user_tmp(self, uid: int) -> dict:
        d = self._tmp.get(uid)
        if d is None:
            d = {}
            self._tmp[uid] = d
        return d

    def _main_text(self):
        s = AccountRepo.get_stats()
        active = sum(1 for o in ORDERS.values() if o.status in (RentStatus.ACTIVE, RentStatus.BUSY))
        return (f"<b>🎮 Настройки Auto Steam Rent</b>\n\n"
                f"∟ Аккаунтов: <code>{s['total']}</code>\n"
                f"∟ Лотов: <code>{len(SETTINGS.lots)}</code>\n"
                f"∟ Активных аренд: <code>{active}</code>\n")

    def _main_kb(self):
        kb = K(row_width=1)
        kb.row(B(f"{_is_on(SETTINGS.enabled)} Авто-выдача", None, f"{CBT.TOGGLE}:enabled"))
        kb.add(B(f"{_is_on(SETTINGS.autoback_on_error)} Авто-возврат", None,
                 f"{CBT.TOGGLE}:autoback_on_error"))
        kb.add(B("📂 Аккаунты", None, CBT.ACC_MENU), B("🔗 Лоты", None, CBT.LOTS))
        kb.add(B("⭐️ Бонусы за отзывы", None, CBT.REVS))
        kb.row(B("🔔 Уведомления", None, CBT.NOTIFS), B("💬 Сообщения", None, CBT.MSGS))
        kb.row(B("📊 Статистика", None, CBT.STATS), B("📜 История", None, f"{CBT.HIST}:1"))
        kb.row(B("🔍 Отладка", None, CBT.DEBUG), B("📁 Файлы", None, f"{CBT.FILES}:all"))
        kb.add(B("⬅️ Назад", None, f"{_CBT.EDIT_PLUGIN}:{UUID}:0"))
        return kb

    def _acc_text(self, acc):
        icon = ICON_STATUS.get(acc.status, "❓")
        lines = [
            f"<b>{icon} Аккаунт #{acc.id}: {acc.login}</b>\n",
            f"∟ Статус: <code>{acc.status}</code>",
            f"∟ Тег: <code>{acc.tag}</code>",
            f"∟ Периоды: <code>{_format_periods(acc.allowed_hours)}</code>",
            f"∟ Пароль: <code>{acc.password}</code>",
        ]
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            lines.append(f"∟ Текущая аренда: <code>{_period_label(acc.rent_hours)}</code>")
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

    def _acc_kb(self, acc):
        kb = K(row_width=2)
        kb.add(B("🔑 Выдать код", None, f"{CBT.ACC_CODE}:{acc.id}"),
               B("🔄 Сменить пароль", None, f"{CBT.ACC_CHPWD}:{acc.id}"))
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            kb.add(B("⏹ Остановить", None, f"{CBT.ACC_STOP}:{acc.id}"),
                   B("⏰ Продлить", None, f"{CBT.ACC_EXTEND}:{acc.id}"))
        if acc.status in (RentStatus.FREE, RentStatus.ERROR):
            kb.add(B("🤝 Ручная аренда", None, f"{CBT.ACC_MANUAL}:{acc.id}"))
        if acc.status == RentStatus.ERROR:
            kb.add(B("🔓 Сброс в FREE", None, f"{CBT.ACC_RESET}:{acc.id}"))
        kb.add(B("⏱ Периоды", None, f"{CBT.ACC_EDIT_HOURS}:{acc.id}"))
        kb.add(B("🗑 Удалить", None, f"{CBT.ACC_DEL}:{acc.id}"))
        kb.add(B("⬅️ К списку", None, f"{CBT.ACC_LIST}:0"))
        return kb

    def open_main(self, c):
        self._edit(c.message, self._main_text(), self._main_kb())

    def open_main_cmd(self, m):
        self._send(m.chat.id, self._main_text(), self._main_kb())

    def toggle_setting(self, c):
        p = self._p(c)
        SETTINGS.toggle(p)
        self.open_notifs(c) if p.startswith("notification") else self.open_main(c)

    def open_acc_menu(self, c):
        kb = K()
        kb.add(B("➕ Добавить аккаунт", None, CBT.ACC_ADD))
        if ACCOUNTS:
            kb.add(B("📜 Список аккаунтов", None, f"{CBT.ACC_LIST}:0"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        self._edit(c.message, "<b>📂 Управление аккаунтами</b>", kb)

    def open_acc_list(self, c):
        pg = self._pid(c)
        kb = K(row_width=1)
        total = len(ACCOUNTS)
        start, end = pg * PAGE_SIZE, (pg + 1) * PAGE_SIZE
        for acc in ACCOUNTS[start:end]:
            icon = ICON_STATUS.get(acc.status, "❓")
            hrs = _format_periods(acc.allowed_hours)
            owner = f' | {acc.owner}' if acc.owner else ''
            kb.add(B(f"{icon} {acc.login} [{acc.tag}] {hrs}{owner}",
                     None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        tp = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        nav = []
        if pg > 0:
            nav.append(B("⬅️", None, f"{CBT.ACC_LIST}:{pg - 1}"))
        nav.append(B(f"{pg + 1}/{tp}", None, _CBT.EMPTY))
        if end < total:
            nav.append(B("➡️", None, f"{CBT.ACC_LIST}:{pg + 1}"))
        if nav:
            kb.row(*nav)
        kb.add(B("⬅️ Назад", None, CBT.ACC_MENU))
        self._edit(c.message, f"<b>📜 Аккаунты ({total})</b>", kb)

    def open_acc_detail(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        self._edit(c.message, self._acc_text(acc), self._acc_kb(acc))

    def acc_code(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        ss = acc.mafile.get("shared_secret", "")
        if not ss:
            return self._answer(c, "❌ Нет shared_secret", True)
        code = SteamGuard.code_sync(ss)
        if code in ("ERROR", "NO_SECRET"):
            return self._answer(c, "❌ Ошибка генерации", True)
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY) and acc.owner_chat_id:
            _send_fp(self.card, acc.owner_chat_id,
                     _tmpl(SETTINGS.messages.guard_code, code=code, end_time=acc.rental_end or "?"))
        kb = K(row_width=2)
        kb.add(B("🔄 Новый код", None, f"{CBT.ACC_CODE}:{acc.id}"),
               B("⬅️ К аккаунту", None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        self._edit(c.message,
                   f"🔑 <b>Steam Guard код</b>\n\n∟ Аккаунт: <code>{acc.login}</code>\n"
                   f"∟ Код: <code>{code}</code>\n∟ Действителен ~30 сек", kb)

    def _run_in_thread(self, c, fn, loading_text, back_target):
        threading.Thread(target=fn, daemon=True).start()
        self._edit(c.message, loading_text, self._back_kb(back_target))

    def acc_stop(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        if acc.status not in (RentStatus.ACTIVE, RentStatus.BUSY):
            return self._answer(c, "ℹ️ Не активна", True)
        order = ORDERS.get(acc.current_order) if acc.current_order else None
        if acc.owner_chat_id:
            _send_fp(self.card, acc.owner_chat_id, SETTINGS.messages.rent_over)
        chat_id = c.message.chat.id

        def _do():
            try:
                _recover_account(self.card, acc, order, "MANUAL_STOP")
                self._send(chat_id, f"✅ Аренда <code>{acc.login}</code> остановлена.")
            except Exception as e:
                self._send(chat_id, f"❌ Ошибка: {e}")

        self._run_in_thread(c, _do, f"⏳ Остановка <code>{acc.login}</code>...",
                            f"{CBT.ACC_DETAIL}:{acc.id}")

    def acc_chpwd(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        if not PLAYWRIGHT_AVAILABLE:
            return self._answer(c, "❌ Playwright не установлен!", True)
        chat_id = c.message.chat.id

        def _do():
            try:
                np = change_password_sync(acc.mafile, acc.password)
                acc.password = np
                storage.mark_dirty("accounts")
                self._send(chat_id, f"✅ Пароль <code>{acc.login}</code>:\n<code>{np}</code>")
            except Exception as e:
                self._send(chat_id, f"❌ Ошибка: {e}")

        self._run_in_thread(c, _do, f"⏳ Смена пароля <code>{acc.login}</code>...",
                            f"{CBT.ACC_DETAIL}:{acc.id}")

    def acc_extend_menu(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        kb = K(row_width=3)
        for h in [1, 2, 3, 6, 12, 24]:
            kb.add(B(f"+{h}ч", None, f"{CBT.ACC_EXTEND_DO}:{acc.id}:{h}"))
        kb.add(B("⬅️", None, f"{CBT.ACC_DETAIL}:{acc.id}"))
        self._edit(c.message, f"⏰ Продлить <code>{acc.login}</code>:", kb)

    def acc_extend_do(self, c):
        parts = c.data.split(":")
        aid, h = int(parts[1]), int(parts[2])
        ne = AccountRepo.extend_rent(aid, h)
        acc = AccountRepo.get(aid)
        if ne:
            if acc and acc.owner_chat_id:
                _send_fp(self.card, acc.owner_chat_id,
                         _tmpl(SETTINGS.messages.extended, hours=str(h), end_time=ne))
            login = acc.login if acc else str(aid)
            self._edit(c.message,
                       f"✅ <code>{login}</code> +{h}ч\n∟ Окончание: <code>{ne}</code>",
                       self._back_kb(f"{CBT.ACC_DETAIL}:{aid}"))
        else:
            self._answer(c, "❌ Не удалось", True)

    def acc_reset(self, c):
        aid = self._pid(c)
        acc = AccountRepo.get(aid)
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        if acc.status != RentStatus.ERROR:
            return self._answer(c, "ℹ️ Не в ERROR", True)
        if acc.current_order:
            order = ORDERS.get(acc.current_order)
            if order and order.status not in (RentStatus.FINISHED, RentStatus.REFUND):
                order.update(status=RentStatus.FINISHED)
        AccountRepo.reset_to_free(aid)
        self._answer(c, f"✅ {acc.login} → FREE")
        acc = AccountRepo.get(aid)
        self._edit(c.message, self._acc_text(acc), self._acc_kb(acc))

    def acc_edit_hours(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        d = self._get_user_tmp(c.from_user.id)
        d["ehrs_id"] = acc.id
        d["sel_hrs"] = list(acc.allowed_hours)
        kb = self._hours_kb(acc.allowed_hours, CBT.ACC_TOGGLE_HOUR,
                            f"{CBT.ACC_SAVE_HOURS}:{acc.id}", f"{CBT.ACC_DETAIL}:{acc.id}")
        self._edit(c.message, f"⏱ <b>Периоды для {acc.login}</b>\n\nВыберите (✅ = вкл):", kb)

    def acc_toggle_hour(self, c):
        h = self._pid(c)
        d = self._get_user_tmp(c.from_user.id)
        sel = d.get("sel_hrs", [])
        if h in sel:
            sel.remove(h)
        else:
            sel.append(h)
        aid = d.get("ehrs_id")
        acc = AccountRepo.get(aid) if aid else None
        kb = self._hours_kb(sel, CBT.ACC_TOGGLE_HOUR,
                            f"{CBT.ACC_SAVE_HOURS}:{aid}", f"{CBT.ACC_DETAIL}:{aid}")
        self._edit(c.message, f"⏱ <b>Периоды для {acc.login if acc else '?'}</b>\n\nВыберите (✅ = вкл):", kb)
        self._answer(c)

    def acc_save_hours(self, c):
        aid = self._pid(c)
        sel = self._get_user_tmp(c.from_user.id).get("sel_hrs", [])
        if not sel:
            return self._answer(c, "❌ Выберите хотя бы один!", True)
        AccountRepo.update_allowed_hours(aid, sel)
        acc = AccountRepo.get(aid)
        if acc:
            self._edit(c.message, self._acc_text(acc), self._acc_kb(acc))
            self._answer(c, "✅ Сохранено!")
        else:
            self._answer(c, "❌ Не найден", True)

    def acc_manual_start(self, c):
        acc = AccountRepo.get(self._pid(c))
        if not acc:
            return self._answer(c, "❌ Не найден", True)
        if acc.status not in (RentStatus.FREE, RentStatus.ERROR):
            return self._answer(c, "ℹ️ Не свободен", True)
        self._get_user_tmp(c.from_user.id)["man_id"] = acc.id
        self._answer(c)
        msg = self._send(c.message.chat.id,
                         f"🤝 Ручная аренда <code>{acc.login}</code>\n\nВведите <b>ник</b>:",
                         self._back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))
        self.tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_MAN_BUYER", {})

    def _h_manual_buyer(self, m):
        self._del_msg(m.chat.id, m.message_id)
        self._get_user_tmp(m.from_user.id)["man_buyer"] = m.text.strip()
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.ACC_MANUAL_HOURS}:{h}"))
        self._send(m.chat.id, "Выберите <b>период</b>:", kb)

    def handle_manual_hours(self, c):
        h = self._pid(c)
        d = self._get_user_tmp(c.from_user.id)
        aid = d.get("man_id")
        if not aid:
            return self._answer(c, "❌ Начните заново", True)
        acc = AccountRepo.manual_assign(aid, d.get("man_buyer", "manual"), h)
        if acc:
            self._edit(c.message,
                       f"✅ <code>{acc.login}</code> → <code>{acc.owner}</code> на {h}ч\n"
                       f"∟ Окончание: <code>{acc.rental_end}</code>",
                       self._back_kb(f"{CBT.ACC_DETAIL}:{aid}"))
        else:
            self._edit(c.message, "❌ Не удалось (занят?)", self._back_kb(f"{CBT.ACC_DETAIL}:{aid}"))

    def start_add(self, c):
        self._answer(c)
        msg = self._send(c.message.chat.id, "1️⃣ Введите <b>логин</b>:", self._back_kb(CBT.ACC_MENU))
        self.tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_LOGIN", {})

    def _h_login(self, m):
        if m.text.startswith("/"):
            return
        self._del_msg(m.chat.id, m.message_id)
        self._tmp[m.from_user.id] = {"login": m.text.strip()}
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        msg = self._send(m.chat.id, "2️⃣ Введите <b>пароль</b>:", self._back_kb(CBT.ACC_MENU))
        self.tg.set_state(m.chat.id, msg.message_id, m.from_user.id, "ASR_PASS", {})

    def _h_pass(self, m):
        self._del_msg(m.chat.id, m.message_id)
        self._get_user_tmp(m.from_user.id)["password"] = m.text.strip()
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        msg = self._send(m.chat.id, "3️⃣ Введите <b>тег</b>:", self._back_kb(CBT.ACC_MENU))
        self.tg.set_state(m.chat.id, msg.message_id, m.from_user.id, "ASR_TAG", {})

    def _h_tag(self, m):
        self._del_msg(m.chat.id, m.message_id)
        d = self._get_user_tmp(m.from_user.id)
        d["tag"] = m.text.strip()
        d["sel_hrs"] = []
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        kb = self._hours_kb([], CBT.HRS_TGL, CBT.HRS_DONE, CBT.ACC_MENU)
        self._send(m.chat.id, "4️⃣ Выберите <b>периоды аренды</b>:", kb)

    def hrs_toggle(self, c):
        h = self._pid(c)
        sel = self._get_user_tmp(c.from_user.id).get("sel_hrs", [])
        if h in sel:
            sel.remove(h)
        else:
            sel.append(h)
        kb = self._hours_kb(sel, CBT.HRS_TGL, CBT.HRS_DONE, CBT.ACC_MENU)
        self._edit(c.message, "4️⃣ Выберите <b>периоды аренды</b>:", kb)
        self._answer(c)

    def hrs_done(self, c):
        d = self._get_user_tmp(c.from_user.id)
        sel = d.get("sel_hrs", [])
        if not sel:
            return self._answer(c, "❌ Выберите хотя бы один!", True)
        d["allowed_hours"] = sorted(sel)
        self._answer(c)
        self._edit(c.message, "5️⃣ Отправьте <b>.maFile</b> (файлом или JSON):")
        self.tg.set_state(c.message.chat.id, c.message.id, c.from_user.id, "ASR_MAFILE", {})

    def _h_mafile(self, m):
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        try:
            if m.document:
                fi = self.bot.get_file(m.document.file_id)
                content = self.bot.download_file(fi.file_path)
                mf = json.loads(content)
            else:
                mf = json.loads(m.text)
            self._del_msg(m.chat.id, m.message_id)
            missing = [f for f in ("shared_secret", "identity_secret", "account_name") if not mf.get(f)]
            if missing:
                raise ValueError(f"Отсутствуют поля: {', '.join(missing)}")
            if "Session" not in mf or "SteamID" not in mf.get("Session", {}):
                raise ValueError("Отсутствует Session.SteamID")
            d = self._get_user_tmp(m.from_user.id)
            ok, txt = AccountRepo.add(d["login"], d["password"], mf, d["tag"],
                                      d.get("allowed_hours", [24]))
            self._send(m.chat.id, f"{'✅' if ok else '❌'} {txt}", self._main_kb())
        except Exception as e:
            logger.error(f"Error adding account: {e}")
            self._send(m.chat.id, f"❌ Ошибка: {e}", self._main_kb())

    def acc_del(self, c):
        parts = c.data.split(":")
        if len(parts) > 1:
            try:
                aid = int(parts[1])
                acc = AccountRepo.get(aid)
                AccountRepo.delete(aid)
                self._answer(c, f"✅ {acc.login if acc else aid} удалён")
                try:
                    fake = type('o', (), {'data': f"{CBT.ACC_LIST}:0", 'message': c.message,
                                          'id': c.id, 'from_user': c.from_user})()
                    self.open_acc_list(fake)
                except Exception:
                    self.open_main(c)
                return
            except Exception:
                pass
        kb = K()
        for acc in ACCOUNTS:
            kb.add(B(f"🗑 {acc.login} [{acc.tag}]", None, f"{CBT.ACC_DEL}:{acc.id}"))
        kb.add(B("⬅️ Назад", None, CBT.ACC_MENU))
        self._edit(c.message, "<b>🗑 Удалить аккаунт</b>", kb)

    def open_lots(self, c):
        kb = K()
        for lid in SETTINGS.lots:
            lc = SETTINGS.get_lot(lid)
            if lc:
                kb.add(B(f"🔗 {lid} → {lc.tag} | {_period_label(lc.hours)}",
                         None, f"{CBT.LOT_DEL}:{lid}"))
        kb.add(B("➕ Добавить", None, CBT.LOT_ADD), B("⬅️ Назад", None, CBT.MAIN))
        self._edit(c.message, "<b>🔗 Привязка лотов</b>\n\nНажмите для удаления.", kb)

    def lot_del(self, c):
        SETTINGS.del_lot(self._p(c))
        self.open_lots(c)

    def lot_add(self, c):
        self._answer(c)
        msg = self._send(c.message.chat.id, "Введите <b>ID лота</b>:", self._back_kb(CBT.LOTS))
        self.tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_LOT_ID", {})

    def _h_lot_id(self, m):
        self._del_msg(m.chat.id, m.message_id)
        self._tmp[m.from_user.id] = {"lot_id": m.text.strip()}
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        tags = AccountRepo.all_tags()
        if not tags:
            self._send(m.chat.id, "❌ Сначала добавьте аккаунты!", self._main_kb())
            return
        kb = K()
        for tag in tags:
            kb.add(B(tag, None, f"{CBT.LOT_TAG}:{tag}"))
        kb.add(B("⬅️ Назад", None, CBT.LOTS))
        self._send(m.chat.id, "Выберите <b>тег</b>:", kb)

    def lot_tag(self, c):
        tag = _ntag(self._p(c))
        self._get_user_tmp(c.from_user.id)["lot_tag"] = tag
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.LOT_HRS}:{h}"))
        kb.add(B("⬅️", None, CBT.LOTS))
        self._edit(c.message, f"Тег: <code>{tag}</code>\n\nВыберите <b>период</b>:", kb)

    def lot_hours(self, c):
        h = self._pid(c)
        d = self._get_user_tmp(c.from_user.id)
        lid, tag = d.get("lot_id"), d.get("lot_tag", "default")
        if lid:
            SETTINGS.set_lot(str(lid), tag, h)
        self._edit(c.message,
                   f"✅ Лот {lid} → <code>{tag}</code>, <code>{_period_label(h)}</code>",
                   self._main_kb())

    def open_reviews(self, c):
        rules = SETTINGS.get_review_rules()
        kb = K(row_width=1)
        for r in rules:
            rl = _period_label(r.rent_hours)
            bl = (_period_label(int(r.bonus_hours))
                  if r.bonus_hours == int(r.bonus_hours)
                  else f"{r.bonus_hours}ч")
            kb.add(B(f"🎁 {rl} → +{bl}  ❌", None, f"{CBT.REV_DEL}:{r.rent_hours}"))
        kb.add(B("➕ Добавить", None, CBT.REV_ADD))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        txt = "<b>⭐️ Бонусы за отзывы</b>\n\n"
        if rules:
            txt += "".join(f"∟ от <code>{_period_label(r.rent_hours)}</code> → "
                           f"<code>+{r.bonus_hours}ч</code>\n" for r in rules)
            txt += "\nНажмите для удаления."
        else:
            txt += "Правил нет."
        self._edit(c.message, txt, kb)

    def rev_add(self, c):
        self._answer(c)
        kb = K(row_width=3)
        for h in ALL_PERIODS:
            kb.add(B(_period_label(h), None, f"{CBT.REV_HRS}:{h}"))
        kb.add(B("⬅️", None, CBT.REVS))
        self._edit(c.message, "Мин. <b>период аренды</b>:", kb)

    def rev_hours(self, c):
        h = self._pid(c)
        self._get_user_tmp(c.from_user.id)["rev_rh"] = h
        kb = K(row_width=3)
        for bh in [1, 2, 3, 6, 12, 24]:
            kb.add(B(_period_label(bh), None, f"{CBT.REV_BON}:{bh}"))
        kb.add(B("⬅️", None, CBT.REVS))
        self._edit(c.message, f"Аренда от: <code>{_period_label(h)}</code>\n\n<b>Бонус</b>:", kb)

    def rev_bonus(self, c):
        bh = self._pid(c)
        rh = self._get_user_tmp(c.from_user.id).get("rev_rh", 3)
        SETTINGS.add_review_rule(rh, float(bh))
        self._answer(c, f"✅ {_period_label(rh)} → +{_period_label(bh)}")
        self.open_reviews(c)

    def rev_del(self, c):
        SETTINGS.del_review_rule(self._pid(c))
        self.open_reviews(c)

    def open_notifs(self, c):
        kb = K(row_width=1)
        for attr, label in [("notification_order_completed", "Выдача"),
                            ("notification_error", "Ошибки"),
                            ("notification_refund", "Возвраты")]:
            kb.add(B(f"{_is_on(getattr(SETTINGS, attr))} {label}", None, f"{CBT.TOGGLE}:{attr}"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        self._edit(c.message, "<b>🔔 Уведомления</b>", kb)

    def open_msgs(self, c):
        kb = K(row_width=1)
        for key, desc in MessagesConfig.DESCRIPTIONS.items():
            kb.add(B(desc, None, f"{CBT.MSG_EDIT}:{key}"))
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        self._edit(c.message, "<b>💬 Тексты сообщений</b>", kb)

    def msg_edit(self, c):
        key = self._p(c)
        self._get_user_tmp(c.from_user.id)["edit_key"] = key
        self._answer(c)
        cur = getattr(SETTINGS.messages, key, "")
        desc = MessagesConfig.DESCRIPTIONS.get(key, "")
        txt = (f"Описание: {desc}\nТекущий: <code>{cur}</code>\n\n"
               f"Переменные: <code>$login, $password, $rent_period, $code, "
               f"$end_time, $hours, $remaining, $id</code>\n\nВведите новый текст:")
        msg = self._send(c.message.chat.id, txt, self._back_kb(CBT.MSGS))
        self.tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_MSG_EDIT", {})

    def _h_msg_edit(self, m):
        self._del_msg(m.chat.id, m.message_id)
        self.tg.clear_state(m.chat.id, m.from_user.id, True)
        key = self._get_user_tmp(m.from_user.id).get("edit_key")
        if key:
            SETTINGS.set_message(key, m.text.strip())
        self._send(m.chat.id, "✅ Сохранено!", self._main_kb())

    def open_stats(self, c):
        s = AccountRepo.get_stats()
        done = sum(1 for o in ORDERS.values() if o.status == RentStatus.FINISHED)
        refs = sum(1 for o in ORDERS.values() if o.status == RentStatus.REFUND)
        self._edit(c.message,
                   f"<b>📊 Статистика</b>\n\nАккаунтов: {s['total']} | Свободно: {s[RentStatus.FREE]} | "
                   f"Ошибки: {s[RentStatus.ERROR]}\nЗавершено: {done} | Возвратов: {refs}",
                   self._back_kb())

    def open_history(self, c):
        page = self._pid(c)
        items = list(ORDERS.values())[::-1]
        total, per = len(items), 10
        pages = max(1, (total + per - 1) // per)
        page = min(max(1, page), pages)
        sl = items[(page - 1) * per:page * per]
        kb = K()
        _icon_map = {RentStatus.FINISHED: '✅', RentStatus.REFUND: '💰'}
        for o in sl:
            icon = _icon_map.get(o.status, '⏳')
            kb.add(B(f"{icon} #{o.id[:8]}... | {o.buyer}", None, _CBT.EMPTY))
        if pages > 1:
            nav = []
            if page > 1:
                nav.append(B("⬅️", None, f"{CBT.HIST}:{page - 1}"))
            nav.append(B(f"{page}/{pages}", None, _CBT.EMPTY))
            if page < pages:
                nav.append(B("➡️", None, f"{CBT.HIST}:{page + 1}"))
            kb.row(*nav)
        kb.add(B("⬅️ Назад", None, CBT.MAIN))
        self._edit(c.message, f"<b>📜 История ({total})</b>", kb)

    def open_debug(self, c):
        lines = [
            f"<b>🔍 Отладка</b>\n",
            f"Steam offset: {SteamGuard._time_offset}s",
            f"Last sync: {SteamGuard._last_sync}",
            f"Playwright: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}",
            f"Cooldowns: {len(_code_cooldowns)}",
            f"Processed orders: {len(_processed_orders)}",
            f"\n<b>Аккаунты:</b>"
        ]
        for a in ACCOUNTS[:10]:
            lines.append(f"• {a.login}: tag='{a.tag}' status={a.status} "
                         f"hrs={_format_periods(a.allowed_hours)}")
        lines.append("\n<b>Лоты:</b>")
        for lid in list(SETTINGS.lots.keys())[:10]:
            lc = SETTINGS.get_lot(lid)
            if lc:
                lines.append(f"• {lid}: '{lc.tag}' → {_period_label(lc.hours)}")
        lines.append("\n<b>Заказы (5):</b>")
        for oid, o in list(ORDERS.items())[-5:]:
            lines.append(f"• {oid[:10]}... buyer_id={o.buyer_id} status={o.status}")
        self._edit(c.message, "\n".join(lines), self._back_kb())

    def get_files(self, c):
        for f in ("settings.json", "accounts.json", "orders.json"):
            p = _get_path(f)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        self.bot.send_document(c.message.chat.id, fh)
                except Exception:
                    pass
        self._answer(c)

    def register_all(self):
        tg = self.tg

        exact_handlers = {
            CBT.MAIN: self.open_main,
            CBT.ACC_MENU: self.open_acc_menu,
            CBT.ACC_ADD: self.start_add,
            CBT.LOTS: self.open_lots,
            CBT.LOT_ADD: self.lot_add,
            CBT.REVS: self.open_reviews,
            CBT.REV_ADD: self.rev_add,
            CBT.NOTIFS: self.open_notifs,
            CBT.MSGS: self.open_msgs,
            CBT.STATS: self.open_stats,
            CBT.DEBUG: self.open_debug,
            CBT.HRS_DONE: self.hrs_done,
        }
        for key, handler in exact_handlers.items():
            tg.cbq_handler(handler, lambda c, k=key: c.data == k)

        prefix_handlers = {
            CBT.ACC_LIST: self.open_acc_list,
            CBT.ACC_DETAIL: self.open_acc_detail,
            CBT.ACC_CODE: self.acc_code,
            CBT.ACC_STOP: self.acc_stop,
            CBT.ACC_CHPWD: self.acc_chpwd,
            CBT.ACC_EXTEND_DO: self.acc_extend_do,
            CBT.ACC_RESET: self.acc_reset,
            CBT.ACC_EDIT_HOURS: self.acc_edit_hours,
            CBT.ACC_TOGGLE_HOUR: self.acc_toggle_hour,
            CBT.ACC_SAVE_HOURS: self.acc_save_hours,
            CBT.ACC_MANUAL: self.acc_manual_start,
            CBT.ACC_MANUAL_HOURS: self.handle_manual_hours,
            CBT.ACC_DEL: self.acc_del,
            CBT.LOT_TAG: self.lot_tag,
            CBT.LOT_HRS: self.lot_hours,
            CBT.LOT_DEL: self.lot_del,
            CBT.REV_HRS: self.rev_hours,
            CBT.REV_BON: self.rev_bonus,
            CBT.REV_DEL: self.rev_del,
            CBT.MSG_EDIT: self.msg_edit,
            CBT.TOGGLE: self.toggle_setting,
            CBT.HIST: self.open_history,
            CBT.HRS_TGL: self.hrs_toggle,
            CBT.FILES: self.get_files,
        }
        for pfx, handler in prefix_handlers.items():
            tg.cbq_handler(handler, lambda c, p=pfx: c.data.startswith(f"{p}:"))

        tg.cbq_handler(self.acc_extend_menu,
                       lambda c: c.data.startswith(f"{CBT.ACC_EXTEND}:") and c.data.count(":") == 1)
        tg.cbq_handler(self.open_main, lambda c: c.data.startswith(CBT.SP))

        for state, handler in [
            ("ASR_LOGIN", self._h_login), ("ASR_PASS", self._h_pass),
            ("ASR_TAG", self._h_tag), ("ASR_MAN_BUYER", self._h_manual_buyer),
            ("ASR_LOT_ID", self._h_lot_id), ("ASR_MSG_EDIT", self._h_msg_edit),
        ]:
            tg.msg_handler(handler,
                           func=lambda m, s=state: tg.check_state(m.chat.id, m.from_user.id, s))

        self.bot.register_message_handler(
            self._h_mafile, content_types=['document', 'text'],
            func=lambda m: tg.check_state(m.chat.id, m.from_user.id, "ASR_MAFILE"))

        tg.msg_handler(self.open_main_cmd, commands=['auto_steam_rent'])


def init(card: Cardinal):
    global cardinal_ref, tg_logs
    cardinal_ref = card
    tg_logs = TgLogs(card)
    SteamGuard.sync_time_sync()
    if card.telegram:
        ui = TelegramUI(card)
        ui.register_all()
        card.add_telegram_commands(UUID, [
            ("auto_steam_rent", "открыть настройки авто аренды аккаунтов", True),
        ])
    threading.Thread(target=rental_check_loop, args=(card,), daemon=True).start()


BIND_TO_PRE_INIT = [init]
BIND_TO_NEW_ORDER = [process_new_order]
BIND_TO_NEW_MESSAGE = [process_message]
BIND_TO_ORDER_STATUS_CHANGED = [process_order_status_changed]
BIND_TO_DELETE = None