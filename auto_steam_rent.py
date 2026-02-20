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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha1
from typing import Any, ClassVar, Dict, List, Optional, Set
from collections import defaultdict

for pkg, imp in [("aiohttp", "aiohttp"), ("pytz", "pytz"), ("pydantic==1.10.18", "pydantic"),
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

try:
    import pydantic
    if int(pydantic.VERSION.split('.')[0]) >= 2:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pydantic==1.10.18", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        importlib.reload(pydantic)
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
import rsa
from pytz import timezone
from pydantic import BaseModel
from pysteamauth.auth import Steam as _BaseSteam

from cardinal import Cardinal
from FunPayAPI.common.enums import OrderStatuses, MessageTypes
from FunPayAPI.updater.events import NewOrderEvent, NewMessageEvent, OrderStatusChangedEvent
from tg_bot import CBT as _CBT
from telebot.types import InlineKeyboardMarkup as K, InlineKeyboardButton as B

NAME = "Auto Steam Rent"
VERSION = "0.2.0"
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
PASSWORD_CHANGE_TIMEOUT = 180

_CMD_CODE = frozenset(("!steamguard", "!code", "/code", "код", "code"))
_CMD_TIME = frozenset(("!time", "/time", "время", "time"))


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
    def _generate(cls, shared_secret: str) -> str:
        ts = int(time.time()) + cls._time_offset
        tw = int(ts / 30)
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
    return base64.b64encode(
        hmac.new(sb, struct.pack(">Q", timestamp) + tag.encode('utf-8'), sha1).digest()
    ).decode('utf-8')


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


class PasswordChangeParams:
    def __init__(self, s, account, reset, issueid, lost=0, **kwargs):
        self.s = int(s)
        self.account = int(account)
        self.reset = int(reset)
        self.issueid = int(issueid)
        self.lost = int(lost)


class SteamPasswordChanger:
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def __init__(self, mafile: dict, current_password: str):
        self.mafile = mafile
        self.current_password = current_password
        self.login = mafile.get("account_name", "")
        self.shared_secret = mafile.get("shared_secret", "")
        self.identity_secret = mafile.get("identity_secret", "")
        self.device_id = mafile.get("device_id", "")
        self.steamid = int(mafile.get("Session", {}).get("SteamID", 0))
        self._steam: Optional[CustomSteam] = None

    async def change_password(self) -> str:
        if not PLAYWRIGHT_AVAILABLE:
            raise Exception("Playwright не установлен!")
        new_password = _gen_password(20)
        self._steam = CustomSteam(
            login=self.login, password=self.current_password,
            shared_secret=self.shared_secret, identity_secret=self.identity_secret,
            device_id=self.device_id, steamid=self.steamid)
        await self._do_login()
        params = await self._receive_params()
        await self._playwright_trigger(params)
        await asyncio.sleep(3)
        confirmed = await self._confirm_recovery(params)
        if not confirmed:
            raise Exception("Mobile confirmation failed")
        await self._poll_recovery(params)
        await self._verify_recovery_code(params)
        await self._get_next_step(params)
        key = await self._get_rsa_key()
        enc_old = self._encrypt(self.current_password, key["publickey_mod"], key["publickey_exp"])
        await self._verify_password(params, enc_old, key["timestamp"])
        key = await self._get_rsa_key()
        await self._check_password_available(new_password)
        enc_new = self._encrypt(new_password, key["publickey_mod"], key["publickey_exp"])
        await self._change_password_request(params, enc_new, key["timestamp"])
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

    async def _receive_params(self) -> PasswordChangeParams:
        from yarl import URL as YarlURL
        resp = await self._steam.raw_request(
            method="GET",
            url="https://help.steampowered.com/wizard/HelpChangePassword?redir=store/account/",
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://store.steampowered.com/",
                "User-Agent": self.UA,
            },
            allow_redirects=True)
        if resp.history:
            try:
                return PasswordChangeParams(**dict(YarlURL(resp.real_url).query))
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

    async def _playwright_trigger(self, params: PasswordChangeParams):
        cookies_for_pw = []
        for domain in ["help.steampowered.com", "store.steampowered.com", "steamcommunity.com"]:
            try:
                domain_cookies = await self._steam.cookies(domain)
                for name, value in domain_cookies.items():
                    cookies_for_pw.append({"name": name, "value": value,
                                           "domain": f".{domain}", "path": "/"})
            except Exception:
                pass
        url = (f"https://help.steampowered.com/en/wizard/HelpWithLoginInfoEnterCode"
               f"?s={params.s}&account={params.account}&reset={params.reset}"
               f"&lost={params.lost}&issueid={params.issueid}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox",
                      "--disable-dev-shm-usage", "--disable-gpu"])
            context = await browser.new_context(
                user_agent=self.UA, locale="en-US",
                viewport={"width": 1280, "height": 720})
            if cookies_for_pw:
                await context.add_cookies(cookies_for_pw)
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(5)
            except Exception:
                pass
            finally:
                await browser.close()

    async def _confirm_recovery(self, params: PasswordChangeParams) -> bool:
        creator_id = params.s
        try:
            from steamlib.api.trade import SteamTrade
            from steamlib.api.trade.exceptions import NotFoundMobileConfirmationError
            st = SteamTrade(self._steam)
            for attempt in range(10):
                try:
                    if await st.mobile_confirm_by_creator_id(creator_id):
                        return True
                except NotFoundMobileConfirmationError:
                    pass
                except Exception:
                    break
                await asyncio.sleep(3)
        except ImportError:
            pass
        return await self._manual_confirm(creator_id)

    async def _manual_confirm(self, creator_id) -> bool:
        cid_str = str(creator_id)
        cookies = {}
        for domain in ("store.steampowered.com", "help.steampowered.com", "steamcommunity.com"):
            try:
                for n, v in (await self._steam.cookies(domain)).items():
                    cookies[n] = v
            except Exception:
                pass
        async with aiohttp.ClientSession() as session:
            for n, v in cookies.items():
                session.cookie_jar.update_cookies({n: v})
            for attempt in range(15):
                try:
                    await SteamGuard.sync_time_async()
                    ts = int(time.time()) + SteamGuard._time_offset
                    conf_key = _generate_confirmation_key(self.identity_secret, ts, "getlist")
                    async with session.get(
                        "https://steamcommunity.com/mobileconf/getlist",
                        params={"p": self.device_id, "a": str(self.steamid),
                                "k": conf_key, "t": str(ts),
                                "m": "react", "tag": "getlist"}) as resp:
                        try:
                            data = await resp.json()
                        except Exception:
                            await asyncio.sleep(3)
                            continue
                        if not data.get("success"):
                            await asyncio.sleep(3)
                            continue
                        for c in data.get("conf", []):
                            if str(c.get("creator_id", "")) == cid_str:
                                ts2 = int(time.time()) + SteamGuard._time_offset
                                allow_key = _generate_confirmation_key(
                                    self.identity_secret, ts2, "allow")
                                async with session.get(
                                    "https://steamcommunity.com/mobileconf/ajaxop",
                                    params={"p": self.device_id, "a": str(self.steamid),
                                            "k": allow_key, "t": str(ts2),
                                            "m": "react", "tag": "allow",
                                            "op": "allow",
                                            "cid": str(c["id"]),
                                            "ck": c["nonce"]}) as cr:
                                    result = await cr.json()
                                    if result.get("success"):
                                        return True
                except Exception:
                    pass
                await asyncio.sleep(3)
        return False

    async def _poll_recovery(self, params: PasswordChangeParams):
        sid = await self._steam.sessionid("help.steampowered.com")
        for i in range(15):
            try:
                r = await self._steam.json_request(
                    method="POST",
                    url="https://help.steampowered.com/en/wizard/AjaxPollAccountRecoveryConfirmation",
                    data={"sessionid": sid, "wizard_ajax": 1, "s": params.s,
                          "reset": params.reset, "lost": params.lost,
                          "method": 8, "issueid": params.issueid, "gamepad": 0},
                    headers={"Accept": "*/*",
                             "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                             "Origin": "https://help.steampowered.com",
                             "User-Agent": self.UA,
                             "X-Requested-With": "XMLHttpRequest"})
                if r.get("success") or r.get("continue"):
                    return
                if r.get("errorMsg"):
                    raise Exception(r["errorMsg"])
            except Exception as e:
                if "errorMsg" in str(e):
                    raise
            await asyncio.sleep(2)
        raise Exception("Poll confirmation timed out")

    async def _verify_recovery_code(self, params: PasswordChangeParams):
        sid = await self._steam.sessionid("help.steampowered.com")
        r = await self._steam.json_request(
            method="GET",
            url="https://help.steampowered.com/en/wizard/AjaxVerifyAccountRecoveryCode",
            params={"code": "", "s": params.s, "reset": params.reset,
                    "lost": params.lost, "method": 8, "issueid": params.issueid,
                    "sessionid": sid, "wizard_ajax": 1, "gamepad": 0},
            headers={"Accept": "*/*", "User-Agent": self.UA,
                     "X-Requested-With": "XMLHttpRequest"})
        if r.get("errorMsg"):
            raise Exception(r["errorMsg"])

    async def _get_next_step(self, params: PasswordChangeParams):
        sid = await self._steam.sessionid("help.steampowered.com")
        r = await self._steam.json_request(
            method="POST",
            url="https://help.steampowered.com/en/wizard/AjaxAccountRecoveryGetNextStep",
            data={"sessionid": sid, "wizard_ajax": 1, "s": params.s,
                  "account": params.account, "reset": params.reset,
                  "issueid": params.issueid, "lost": 2},
            headers={"Accept": "*/*", "X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Origin": "https://help.steampowered.com", "User-Agent": self.UA})
        if r.get("errorMsg"):
            raise Exception(r["errorMsg"])

    async def _get_rsa_key(self) -> dict:
        sid = await self._steam.sessionid("help.steampowered.com")
        return await self._steam.json_request(
            method="POST",
            url="https://help.steampowered.com/en/login/getrsakey/",
            data={"sessionid": sid, "username": self.login},
            headers={"Accept": "*/*",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Origin": "https://help.steampowered.com",
                     "X-Requested-With": "XMLHttpRequest",
                     "User-Agent": self.UA})

    async def _verify_password(self, params, enc_pwd, ts):
        sid = await self._steam.sessionid("help.steampowered.com")
        r = await self._steam.json_request(
            method="POST",
            url="https://help.steampowered.com/en/wizard/AjaxAccountRecoveryVerifyPassword/",
            data={"sessionid": sid, "s": params.s, "lost": 2, "reset": 1,
                  "password": enc_pwd, "rsatimestamp": ts},
            headers={"Accept": "*/*",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Origin": "https://help.steampowered.com",
                     "X-Requested-With": "XMLHttpRequest", "User-Agent": self.UA})
        if r.get("errorMsg"):
            raise Exception(r["errorMsg"])

    async def _check_password_available(self, password):
        sid = await self._steam.sessionid("help.steampowered.com")
        r = await self._steam.json_request(
            method="POST",
            url="https://help.steampowered.com/en/wizard/AjaxCheckPasswordAvailable/",
            data={"sessionid": sid, "wizard_ajax": 1, "password": password},
            headers={"Origin": "https://help.steampowered.com", "User-Agent": self.UA})
        if not r.get("available"):
            raise Exception("Password not available")

    async def _change_password_request(self, params, enc_pwd, ts):
        sid = await self._steam.sessionid("help.steampowered.com")
        r = await self._steam.json_request(
            method="POST",
            url="https://help.steampowered.com/ru/wizard/AjaxAccountRecoveryChangePassword/",
            data={"sessionid": sid, "wizard_ajax": 1, "s": params.s,
                  "account": params.account, "password": enc_pwd, "rsatimestamp": ts},
            headers={"Accept": "*/*",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Origin": "https://help.steampowered.com",
                     "X-Requested-With": "XMLHttpRequest", "User-Agent": self.UA})
        if r.get("errorMsg"):
            raise Exception(r["errorMsg"])

    @staticmethod
    def _encrypt(password: str, mod: str, exp: str) -> str:
        pk = rsa.PublicKey(n=int(mod, 16), e=int(exp, 16))
        return base64.b64encode(rsa.encrypt(password.encode("ascii"), pk)).decode()


async def change_password_async(mafile: dict, current_password: str) -> str:
    return await SteamPasswordChanger(mafile, current_password).change_password()

def change_password_sync(mafile: dict, current_password: str) -> str:
    new_loop = asyncio.new_event_loop()
    result = [None]
    error = [None]
    def _run():
        try:
            result[0] = new_loop.run_until_complete(change_password_async(mafile, current_password))
        except Exception as e:
            error[0] = e
        finally:
            try:
                pending = asyncio.all_tasks(new_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            new_loop.close()
    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=PASSWORD_CHANGE_TIMEOUT)
    if t.is_alive():
        raise Exception("Password change timed out")
    if error[0]:
        raise error[0]
    if result[0] is None:
        raise Exception("Password change returned no result")
    return result[0]


class LotConfig(BaseModel):
    tag: str
    hours: int
    count: int = 1
    class Config:
        extra = "allow"

class MessagesConfig(BaseModel):
    order_completed: str = ("✅ Данные от аккаунта:\n∟ Логин: $login\n∟ Пароль: $password\n"
                            "∟ Аренда на: $rent_period\n\n⚠️ Для входа вам понадобится Steam Guard код.\n"
                            "Напишите !code чтобы получить код")
    multi_order_completed: str = ("✅ Данные от аккаунтов ($count шт.):\n$accounts_list\n"
                                  "∟ Аренда на: $rent_period\n\n⚠️ Напишите !code чтобы получить коды")
    guard_code: str = "✅ Steam Guard код: $code\n∟ Действителен ~30 секунд\n∟ Аренда до: $end_time"
    multi_guard_code: str = "✅ Steam Guard коды:\n$codes_list\n∟ Действительны ~30 секунд"
    rent_over: str = "⛔ Аренда завершена!\n∟ Пароль изменён"
    warning: str = "⚠️ Аренда заканчивается через 10 минут!"
    extended: str = "✅ Аренда продлена на +$hours ч.\n∟ Окончание: $end_time"
    auto_extended: str = "✅ Аренда автоматически продлена на +$hours ч.\n∟ Окончание: $end_time"
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
        "order_completed": "📋 Выдача данных", "multi_order_completed": "📋 Выдача мульти",
        "guard_code": "🔑 Steam Guard код", "multi_guard_code": "🔑 Мульти коды",
        "rent_over": "⛔ Конец аренды", "warning": "⚠️ Предупреждение 10 мин",
        "extended": "✅ Продление", "auto_extended": "✅ Авто-продление",
        "bonus": "🎁 Бонус за отзыв",
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
    chat_id: Any
    buyer: str
    buyer_id: int
    acc_id: int
    hours: float
    status: str = RentStatus.BUSY
    warned: bool = False
    review_claimed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    acc_ids: List[int] = field(default_factory=list)
    is_multi: bool = False
    is_extension: bool = False
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        _save_orders()
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "chat_id", "buyer", "buyer_id", "acc_id",
            "hours", "status", "warned", "review_claimed", "created_at",
            "acc_ids", "is_multi", "is_extension")}

class Settings(BaseModel):
    enabled: bool = True
    autoback_on_error: bool = True
    auto_extend: bool = False
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
    def get_lot(self, lot_id: str) -> Optional[LotConfig]:
        raw = self.lots.get(str(lot_id))
        if raw is None:
            return None
        if isinstance(raw, str):
            return LotConfig(tag=_ntag(raw), hours=24, count=1)
        if isinstance(raw, dict):
            return LotConfig(**raw)
        return None
    def set_lot(self, lot_id: str, tag: str, hours: int, count: int = 1):
        self.lots[str(lot_id)] = {"tag": _ntag(tag), "hours": hours, "count": count}
        _save_settings()
    def del_lot(self, lot_id: str):
        self.lots.pop(str(lot_id), None)
        _save_settings()
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
_processed_orders: Set[str] = set()
_temp_storage: Dict[int, dict] = {}

def _save_settings():
    _save_json("settings", SETTINGS.dict())
def _save_accounts():
    _save_json("accounts", [a.dict() for a in ACCOUNTS])
def _save_orders():
    _save_json("orders", {k: v.to_dict() for k, v in ORDERS.items()})

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
            SETTINGS.lots[lid] = {"tag": _ntag(val), "hours": 24, "count": 1}
            changed = True
        elif isinstance(val, dict) and "count" not in val:
            val["count"] = 1
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
            v.setdefault("acc_ids", [])
            v.setdefault("is_multi", False)
            v.setdefault("is_extension", False)
        ORDERS = {k: RentOrder(**v) for k, v in d.items()}
    else:
        ORDERS = {}
    _processed_orders.update(ORDERS.keys())

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
        for a in ACCOUNTS:
            if _ntag(a.tag) != tag or a.status != RentStatus.FREE:
                continue
            if hours is not None and hours not in a.allowed_hours:
                continue
            return a
        return None
    @staticmethod
    def get_free_multi(tag: str, hours: int, count: int) -> List[AccountModel]:
        tag = _ntag(tag)
        result = []
        for a in ACCOUNTS:
            if _ntag(a.tag) != tag or a.status != RentStatus.FREE:
                continue
            if hours is not None and hours not in a.allowed_hours:
                continue
            result.append(a)
            if len(result) >= count:
                break
        return result
    @staticmethod
    def add(login, password, mafile, tag, allowed_hours=None):
        if allowed_hours is None:
            allowed_hours = [24]
        tag = _ntag(tag)
        if any(a.login.lower() == login.lower() for a in ACCOUNTS):
            return False, "Аккаунт уже существует"
        nid = max((a.id for a in ACCOUNTS), default=0) + 1
        ACCOUNTS.append(AccountModel(
            id=nid, login=login, password=password, mafile=mafile,
            tag=tag, allowed_hours=sorted(allowed_hours), rent_hours=allowed_hours[0]))
        _save_accounts()
        return True, (f"Аккаунт {login} добавлен (ID: {nid}, тег: {tag}, "
                      f"периоды: {_format_periods(allowed_hours)})")
    @staticmethod
    def delete(acc_id: int) -> bool:
        for i, a in enumerate(ACCOUNTS):
            if a.id == acc_id:
                del ACCOUNTS[i]
                _save_accounts()
                return True
        return False
    @staticmethod
    def assign(acc_id, order_id, buyer, buyer_id, chat_id, hours):
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
        acc = AccountRepo.by_order(order_id)
        if acc:
            acc.status = RentStatus.ACTIVE
            acc.rental_end = _fmt(_now() + timedelta(hours=acc.rent_hours))
            _save_accounts()
        return acc
    @staticmethod
    def start_rent_multi(order_id, acc_ids: List[int]):
        for aid in acc_ids:
            acc = AccountRepo.get(aid)
            if acc and acc.current_order == order_id:
                acc.status = RentStatus.ACTIVE
                acc.rental_end = _fmt(_now() + timedelta(hours=acc.rent_hours))
        _save_accounts()
    @staticmethod
    def extend_rent(acc_id: int, hours: float) -> Optional[str]:
        acc = AccountRepo.get(acc_id)
        if acc and acc.rental_end:
            acc.rental_end = _fmt(_parse(acc.rental_end) + timedelta(hours=hours))
            _save_accounts()
            return acc.rental_end
        return None
    @staticmethod
    def release(acc_id: int, new_password: str = None, error: bool = False):
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
                                acc_id=acc.id, hours=float(hours), status=RentStatus.ACTIVE)
        _save_accounts()
        _save_orders()
        return acc
    @staticmethod
    def update_allowed_hours(acc_id: int, allowed_hours: List[int]) -> bool:
        acc = AccountRepo.get(acc_id)
        if not acc:
            return False
        acc.allowed_hours = sorted(allowed_hours)
        if acc.status == RentStatus.FREE:
            acc.rent_hours = allowed_hours[0] if allowed_hours else 24
        _save_accounts()
        return True
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
            ck = str(o.chat_id) if o.chat_id is not None else ""
            if ck == key:
                return o
            if ck.startswith("users-") and key in ck.split("-")[1:]:
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
    def get_accounts_for_order(order: RentOrder) -> List[AccountModel]:
        if order.is_multi and order.acc_ids:
            return [a for a in ACCOUNTS if a.id in order.acc_ids]
        acc = AccountRepo.get(order.acc_id)
        return [acc] if acc else []


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

def _get_lot_from_description(c, description):
    try:
        for lid, lot_cfg in SETTINGS.lots.items():
            if lot_cfg["tag"] in description.lower() or lot_cfg["hours"] in description:
                return lid
        import re
        lot_id_match = re.search(r'lot_id=(\d+)', description)
        if lot_id_match:
            return lot_id_match.group(1)
    except Exception:
        pass
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
        logger.error(f"Password change failed for {acc.login}: {e}")
        AccountRepo.release(acc.id, error=True)
        if tg_logs:
            tg_logs.error(f"Смена пароля: {acc.login} - {str(e)[:80]}")
        if SETTINGS.autoback_on_error and order:
            if _do_refund(c, order.id):
                order.update(status=RentStatus.REFUND)
                if tg_logs:
                    tg_logs.refund(order.id, f"Ошибка смены пароля: {acc.login}")

def _recover_multi(c, acc_ids: List[int], order, reason):
    for aid in acc_ids:
        acc = AccountRepo.get(aid)
        if acc and acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            _recover_account(c, acc, None, reason)
    if order:
        order.update(status=RentStatus.FINISHED)
        if reason == "TIME" and order.chat_id:
            _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.rent_over, id=order.id))


def process_new_order(c, event):
    if not SETTINGS or not SETTINGS.enabled:
        return
    order = event.order
    if not order:
        return
    order_id = getattr(order, 'id', None)
    if not order_id or order_id in _processed_orders or order_id in ORDERS:
        return
    _processed_orders.add(order_id)
    desc = getattr(order, 'description', '') or ''
    lot_id = _get_lot_from_description(c, desc)
    if lot_id is not None:
        lot_id = str(lot_id)
    if not lot_id:
        return
    lot_cfg = SETTINGS.get_lot(lot_id)
    if not lot_cfg:
        return
    tag, hours, count = _ntag(lot_cfg.tag), lot_cfg.hours, lot_cfg.count
    if hours <= 0:
        hours = 24
    if count <= 0:
        count = 1

    if SETTINGS.auto_extend and count == 1:
        existing = AccountRepo.find_active_by_buyer(order.buyer_id, tag)
        if existing:
            acc = AccountRepo.get(existing.acc_id)
            if acc:
                ne = AccountRepo.extend_rent(acc.id, hours)
                if ne:
                    ext_order = RentOrder(
                        id=order.id, chat_id=order.chat_id, buyer=order.buyer_username,
                        buyer_id=order.buyer_id, acc_id=acc.id, hours=float(hours),
                        status=RentStatus.ACTIVE, is_extension=True)
                    ORDERS[order.id] = ext_order
                    _save_orders()
                    _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.auto_extended,
                                                      hours=str(hours), end_time=ne))
                    logger.info(f"Auto-extended {acc.login} for {order.buyer_username} +{hours}h")
                    return

    if count > 1:
        _process_multi_order(c, order, tag, hours, count)
    else:
        _process_single_order(c, order, tag, hours)

def _process_single_order(c, order, tag, hours):
    acc = AccountRepo.get_free(tag, hours)
    if not acc:
        if SETTINGS.autoback_on_error:
            _send_fp(c, order.chat_id, SETTINGS.messages.no_accounts)
            if _do_refund(c, order.id):
                _send_fp(c, order.chat_id, SETTINGS.messages.refunded)
                if tg_logs:
                    tg_logs.refund(order.id, f"Нет аккаунтов (тег: {tag})")
        return
    AccountRepo.assign(acc.id, order.id, order.buyer_username, order.buyer_id, order.chat_id, hours)
    ro = RentOrder(id=order.id, chat_id=order.chat_id, buyer=order.buyer_username,
                   buyer_id=order.buyer_id, acc_id=acc.id, hours=float(hours))
    ORDERS[order.id] = ro
    _save_orders()
    _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.order_completed,
                                      login=acc.login, password=acc.password, id=order.id,
                                      rent_period=_period_label(hours)))
    if tg_logs:
        tg_logs.order_completed(ro, acc.login)

def _process_multi_order(c, order, tag, hours, count):
    accs = AccountRepo.get_free_multi(tag, hours, count)
    if len(accs) < count:
        if SETTINGS.autoback_on_error:
            _send_fp(c, order.chat_id, f"❌ Нужно {count} аккаунтов, свободно {len(accs)}. Средства будут возвращены.")
            if _do_refund(c, order.id):
                _send_fp(c, order.chat_id, SETTINGS.messages.refunded)
                if tg_logs:
                    tg_logs.refund(order.id, f"Недостаточно аккаунтов: {len(accs)}/{count}")
        return

    acc_ids = []
    accounts_list = []
    for acc in accs:
        AccountRepo.assign(acc.id, order.id, order.buyer_username, order.buyer_id, order.chat_id, hours)
        acc_ids.append(acc.id)
        accounts_list.append(f"∟ Логин: {acc.login} | Пароль: {acc.password}")

    ro = RentOrder(id=order.id, chat_id=order.chat_id, buyer=order.buyer_username,
                   buyer_id=order.buyer_id, acc_id=accs[0].id, hours=float(hours),
                   acc_ids=acc_ids, is_multi=True)
    ORDERS[order.id] = ro
    _save_orders()

    _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.multi_order_completed,
                                      count=str(count),
                                      accounts_list="\n".join(accounts_list),
                                      rent_period=_period_label(hours)))
    if tg_logs:
        logins = ", ".join(a.login for a in accs)
        tg_logs.order_completed(ro, logins)


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

    accs = AccountRepo.get_accounts_for_order(order)
    if not accs:
        _send_fp(c, msg.chat_id, SETTINGS.messages.no_account)
        return

    if order.chat_id != msg.chat_id:
        order.chat_id = msg.chat_id
        _save_orders()
    for acc in accs:
        if acc.owner_chat_id != msg.chat_id:
            acc.owner_chat_id = msg.chat_id
            _save_accounts()

    if is_code:
        cd_key = str(msg.chat_id)
        now_ts = time.time()
        last = _code_cooldowns.get(cd_key)
        if last is not None and now_ts - last < CODE_COOLDOWN:
            return
        _code_cooldowns[cd_key] = now_ts

        if order.is_multi and len(accs) > 1:
            _handle_multi_code(c, msg, order, accs)
        else:
            _handle_single_code(c, msg, order, accs[0])

    elif is_time:
        acc = accs[0]
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

def _handle_single_code(c, msg, order, acc):
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
    _save_accounts()

def _handle_multi_code(c, msg, order, accs):
    if order.status == RentStatus.BUSY:
        AccountRepo.start_rent_multi(order.id, order.acc_ids)
        order.update(status=RentStatus.ACTIVE)

    codes_lines = []
    for acc in accs:
        ss = acc.mafile.get("shared_secret", "")
        if not ss:
            codes_lines.append(f"∟ {acc.login}: ❌ нет секрета")
            continue
        code = SteamGuard.code_sync(ss)
        if code in ("ERROR", "NO_SECRET"):
            codes_lines.append(f"∟ {acc.login}: ❌ ошибка")
        else:
            codes_lines.append(f"∟ {acc.login}: {code}")
        acc.access_count += 1

    _save_accounts()
    _send_fp(c, msg.chat_id, _tmpl(SETTINGS.messages.multi_guard_code,
                                    codes_list="\n".join(codes_lines)))

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
        if order.is_multi and order.acc_ids:
            for aid in order.acc_ids:
                AccountRepo.extend_rent(aid, bonus)
            order.update(review_claimed=True)
            _send_fp(c, order.chat_id, _tmpl(SETTINGS.messages.bonus, hours=str(bonus)))
        else:
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
        if order.is_multi and order.acc_ids:
            _recover_multi(c, order.acc_ids, order, "REFUND_EXT")
        else:
            acc = AccountRepo.by_order(event.order.id) or AccountRepo.get(order.acc_id)
            if acc:
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
                            if order.is_multi and order.acc_ids:
                                _recover_multi(c, order.acc_ids, order, "REFUND")
                            else:
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
                        if order.is_multi and order.acc_ids:
                            all_expired = all(
                                AccountRepo.get(aid) is None or
                                not AccountRepo.get(aid).rental_end or
                                (_parse(AccountRepo.get(aid).rental_end) - now).total_seconds() <= 0
                                for aid in order.acc_ids
                            )
                            if all_expired:
                                _recover_multi(c, order.acc_ids, order, "TIME")
                        else:
                            _recover_account(c, acc, order, "TIME")
        except Exception:
            pass
        cleanup_counter += 1
        if cleanup_counter >= 30:
            cleanup_counter = 0
            try:
                cutoff = (_now() - timedelta(days=30)).isoformat()
                old = [k for k, o in ORDERS.items()
                       if o.status in (RentStatus.FINISHED, RentStatus.REFUND) and o.created_at < cutoff]
                for k in old:
                    del ORDERS[k]
                    _processed_orders.discard(k)
                if old:
                    _save_orders()
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
    LOT_CNT = "asr_lcnt"
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


def _try_delete(bot, cid, mid):
    try:
        bot.delete_message(cid, mid)
    except Exception:
        pass


def init(card: Cardinal):
    global cardinal_ref, tg_logs
    cardinal_ref = card
    tg_logs = TgLogs(card)
    SteamGuard.sync_time_sync()
    if not card.telegram:
        threading.Thread(target=rental_check_loop, args=(card,), daemon=True).start()
        return
    tg, bot = card.telegram, card.telegram.bot

    def send(cid, text, kb=None):
        real_id = cid.chat.id if hasattr(cid, 'chat') else cid
        return bot.send_message(real_id, text, reply_markup=kb, parse_mode='HTML')

    def edit(m, text, kb=None):
        try:
            return bot.edit_message_text(text, m.chat.id, m.message_id, reply_markup=kb, parse_mode='HTML')
        except Exception as e:
            if "message is not modified" not in str(e):
                raise

    def answer(cb, msg=None, alert=False):
        return bot.answer_callback_query(cb.id, msg, show_alert=alert)

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
        kb.row(B("🔍 Отладка", None, CBT.DEBUG), B("📁 Файлы", None, f"{CBT.FILES}:all"))
        kb.add(B("⬅️ Назад", None, f"{_CBT.EDIT_PLUGIN}:{UUID}:0"))
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
        if acc.status in (RentStatus.ACTIVE, RentStatus.BUSY):
            kb.add(B("⏹ Остановить", None, f"{CBT.ACC_STOP}:{acc.id}"),
                   B("⏰ Продлить", None, f"{CBT.ACC_EXTEND}:{acc.id}"))
        if acc.status in (RentStatus.FREE, RentStatus.ERROR):
            kb.add(B("🤝 Ручная аренда", None, f"{CBT.ACC_MANUAL}:{acc.id}"))
        if acc.status == RentStatus.ERROR:
            kb.add(B("🔓 Сброс FREE", None, f"{CBT.ACC_RESET}:{acc.id}"))
        kb.add(B("⏱ Периоды", None, f"{CBT.ACC_EDIT_HOURS}:{acc.id}"))
        kb.add(B("🗑 Удалить", None, f"{CBT.ACC_DEL}:{acc.id}"))
        kb.add(B("⬅️ К списку", None, f"{CBT.ACC_LIST}:0"))
        return kb

    def open_main(c):
        edit(c.message, _main_text(), _main_kb())
    def open_main_cmd(m):
        send(m.chat.id, _main_text(), _main_kb())
    def toggle_setting(c):
        p = _p(c)
        SETTINGS.toggle(p)
        open_notifs(c) if p.startswith("notification") else open_main(c)
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
        start, end = pg * PAGE_SIZE, (pg + 1) * PAGE_SIZE
        for acc in ACCOUNTS[start:end]:
            icon = ICON_STATUS.get(acc.status, "❓")
            owner = f' | {acc.owner}' if acc.owner else ''
            kb.add(B(f"{icon} {acc.login} [{acc.tag}] {_format_periods(acc.allowed_hours)}{owner}",
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
        edit(c.message, f"<b>📜 Аккаунты ({total})</b>", kb)
    def open_acc_detail(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        edit(c.message, _acc_text(acc), _acc_kb(acc))
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
        order = ORDERS.get(acc.current_order) if acc.current_order else None
        if acc.owner_chat_id:
            _send_fp(card, acc.owner_chat_id, SETTINGS.messages.rent_over)
        chat_id = c.message.chat.id
        acc_id = acc.id
        def _do():
            try:
                a = AccountRepo.get(acc_id)
                if a:
                    _recover_account(card, a, order, "MANUAL_STOP")
                    send(chat_id, f"✅ Аренда <code>{a.login}</code> остановлена.")
            except Exception as e:
                send(chat_id, f"❌ Ошибка: {e}")
        answer(c)
        edit(c.message, f"⏳ Остановка <code>{acc.login}</code>...", _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))
        threading.Thread(target=_do, daemon=True).start()
    def acc_chpwd(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        if not PLAYWRIGHT_AVAILABLE:
            return answer(c, "❌ Playwright не установлен!", True)
        chat_id = c.message.chat.id
        acc_id = acc.id
        def _do():
            try:
                a = AccountRepo.get(acc_id)
                if not a:
                    send(chat_id, "❌ Аккаунт не найден")
                    return
                np = change_password_sync(a.mafile, a.password)
                a.password = np
                _save_accounts()
                send(chat_id, f"✅ Пароль <code>{a.login}</code>:\n<code>{np}</code>")
            except Exception as e:
                send(chat_id, f"❌ Ошибка: {e}")
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
        parts = c.data.split(":")
        aid, h = int(parts[1]), int(parts[2])
        ne = AccountRepo.extend_rent(aid, h)
        acc = AccountRepo.get(aid)
        if ne:
            if acc and acc.owner_chat_id:
                _send_fp(card, acc.owner_chat_id, _tmpl(SETTINGS.messages.extended, hours=str(h), end_time=ne))
            login = acc.login if acc else str(aid)
            edit(c.message, f"✅ <code>{login}</code> +{h}ч\n∟ Окончание: <code>{ne}</code>",
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
        if acc.current_order:
            order = ORDERS.get(acc.current_order)
            if order and order.status not in (RentStatus.FINISHED, RentStatus.REFUND):
                order.update(status=RentStatus.FINISHED)
        AccountRepo.reset_to_free(aid)
        answer(c, f"✅ {acc.login} → FREE")
        acc = AccountRepo.get(aid)
        edit(c.message, _acc_text(acc), _acc_kb(acc))
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
    def acc_manual_start(c):
        acc = AccountRepo.get(_pid(c))
        if not acc:
            return answer(c, "❌ Не найден", True)
        if acc.status not in (RentStatus.FREE, RentStatus.ERROR):
            return answer(c, "ℹ️ Не свободен", True)
        _temp_storage[c.from_user.id] = {"man_id": acc.id}
        answer(c)
        msg = send(c.message.chat.id, f"🤝 Ручная аренда <code>{acc.login}</code>\n\nВведите <b>ник</b>:",
                   _back_kb(f"{CBT.ACC_DETAIL}:{acc.id}"))
        tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_MAN_BUYER", {})
    def _h_manual_buyer(m):
        _try_delete(bot, m.chat.id, m.message_id)
        _temp_storage.setdefault(m.from_user.id, {})["man_buyer"] = m.text.strip()
        tg.clear_state(m.chat.id, m.from_user.id, True)
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
        msg = send(c.message.chat.id, "1️⃣ Введите <b>логин</b>:", _back_kb(CBT.ACC_MENU))
        tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_LOGIN", {})
    def _h_login(m):
        if m.text.startswith("/"):
            return
        _try_delete(bot, m.chat.id, m.message_id)
        _temp_storage[m.from_user.id] = {"login": m.text.strip()}
        tg.clear_state(m.chat.id, m.from_user.id, True)
        msg = send(m.chat.id, "2️⃣ Введите <b>пароль</b>:", _back_kb(CBT.ACC_MENU))
        tg.set_state(m.chat.id, msg.message_id, m.from_user.id, "ASR_PASS", {})
    def _h_pass(m):
        _try_delete(bot, m.chat.id, m.message_id)
        _temp_storage.setdefault(m.from_user.id, {})["password"] = m.text.strip()
        tg.clear_state(m.chat.id, m.from_user.id, True)
        msg = send(m.chat.id, "3️⃣ Введите <b>тег</b>:", _back_kb(CBT.ACC_MENU))
        tg.set_state(m.chat.id, msg.message_id, m.from_user.id, "ASR_TAG", {})
    def _h_tag(m):
        _try_delete(bot, m.chat.id, m.message_id)
        d = _temp_storage.setdefault(m.from_user.id, {})
        d["tag"] = m.text.strip()
        d["sel_hrs"] = []
        tg.clear_state(m.chat.id, m.from_user.id, True)
        kb = _hours_kb([], CBT.HRS_TGL, CBT.HRS_DONE, CBT.ACC_MENU)
        send(m.chat.id, "4️⃣ Выберите <b>периоды аренды</b>:", kb)
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
        edit(c.message, "5️⃣ Отправьте <b>.maFile</b> (файлом или JSON):")
        tg.set_state(c.message.chat.id, c.message.id, c.from_user.id, "ASR_MAFILE", {})
    def _h_mafile(m):
        tg.clear_state(m.chat.id, m.from_user.id, True)
        try:
            if m.document:
                content = bot.download_file(bot.get_file(m.document.file_id).file_path).decode('utf-8')
            else:
                content = m.text
            _try_delete(bot, m.chat.id, m.message_id)
            mf = json.loads(content)
            if "shared_secret" not in mf:
                raise ValueError("Отсутствует shared_secret!")
            d = _temp_storage.get(m.from_user.id, {})
            ok, txt = AccountRepo.add(d["login"], d["password"], mf, d["tag"],
                                      d.get("allowed_hours", [24]))
            send(m.chat.id, f"{'✅' if ok else '❌'} {txt}", _main_kb())
        except Exception as e:
            send(m.chat.id, f"❌ Ошибка: {e}", _main_kb())
    def acc_del(c):
        parts = c.data.split(":")
        if len(parts) > 1:
            try:
                aid = int(parts[1])
                acc = AccountRepo.get(aid)
                AccountRepo.delete(aid)
                answer(c, f"✅ {acc.login if acc else aid} удалён")
                try:
                    fake = type('o', (), {'data': f"{CBT.ACC_LIST}:0", 'message': c.message,
                                          'id': c.id, 'from_user': c.from_user})()
                    open_acc_list(fake)
                except Exception:
                    open_main(c)
                return
            except Exception:
                pass
        kb = K()
        for acc in ACCOUNTS:
            kb.add(B(f"🗑 {acc.login} [{acc.tag}]", None, f"{CBT.ACC_DEL}:{acc.id}"))
        kb.add(B("⬅️ Назад", None, CBT.ACC_MENU))
        edit(c.message, "<b>🗑 Удалить аккаунт</b>", kb)
    def open_lots(c):
        kb = K()
        for lid in SETTINGS.lots:
            lc = SETTINGS.get_lot(lid)
            if lc:
                cnt = f" x{lc.count}" if lc.count > 1 else ""
                kb.add(B(f"🔗 {lid} → {lc.tag} | {_period_label(lc.hours)}{cnt}",
                         None, f"{CBT.LOT_DEL}:{lid}"))
        kb.add(B("➕ Добавить", None, CBT.LOT_ADD), B("⬅️ Назад", None, CBT.MAIN))
        edit(c.message, "<b>🔗 Привязка лотов</b>\n\nНажмите для удаления.", kb)
    def lot_del(c):
        SETTINGS.del_lot(_p(c))
        open_lots(c)
    def lot_add(c):
        answer(c)
        msg = send(c.message.chat.id, "Введите <b>ID лота</b>:", _back_kb(CBT.LOTS))
        tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_LOT_ID", {})
    def _h_lot_id(m):
        _try_delete(bot, m.chat.id, m.message_id)
        _temp_storage[m.from_user.id] = {"lot_id": m.text.strip()}
        tg.clear_state(m.chat.id, m.from_user.id, True)
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
        _temp_storage.setdefault(c.from_user.id, {})["lot_hours"] = h
        kb = K(row_width=4)
        for cnt in [1, 2, 3, 4, 5]:
            kb.add(B(f"{cnt} шт.", None, f"{CBT.LOT_CNT}:{cnt}"))
        kb.add(B("⬅️", None, CBT.LOTS))
        edit(c.message, f"Период: <code>{_period_label(h)}</code>\n\n"
                        f"Сколько <b>аккаунтов</b> выдавать за один заказ?", kb)
    def lot_count(c):
        cnt = _pid(c)
        d = _temp_storage.get(c.from_user.id, {})
        lid = d.get("lot_id")
        tag = d.get("lot_tag", "default")
        h = d.get("lot_hours", 24)
        if lid:
            SETTINGS.set_lot(str(lid), tag, h, cnt)
        multi = f" x{cnt}" if cnt > 1 else ""
        edit(c.message, f"✅ Лот {lid} → <code>{tag}</code>, <code>{_period_label(h)}</code>{multi}",
             _main_kb())
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
            txt += "".join(f"∟ от <code>{_period_label(r.rent_hours)}</code> → <code>+{r.bonus_hours}ч</code>\n" for r in rules)
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
                            ("notification_error", "Ошибки"), ("notification_refund", "Возвраты")]:
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
        txt = (f"Описание: {desc}\nТекущий: <code>{cur}</code>\n\n"
               f"Переменные: <code>$login, $password, $rent_period, $code, "
               f"$end_time, $hours, $remaining, $id, $count, $accounts_list, $codes_list</code>\n\nВведите новый текст:")
        msg = send(c.message.chat.id, txt, _back_kb(CBT.MSGS))
        tg.set_state(c.message.chat.id, msg.message_id, c.from_user.id, "ASR_MSG_EDIT", {})
    def _h_msg_edit(m):
        _try_delete(bot, m.chat.id, m.message_id)
        tg.clear_state(m.chat.id, m.from_user.id, True)
        key = _temp_storage.get(m.from_user.id, {}).get("edit_key")
        if key:
            SETTINGS.set_message(key, m.text.strip())
        send(m.chat.id, "✅ Сохранено!", _main_kb())
    def open_stats(c):
        s = AccountRepo.get_stats()
        done = sum(1 for o in ORDERS.values() if o.status == RentStatus.FINISHED)
        refs = sum(1 for o in ORDERS.values() if o.status == RentStatus.REFUND)
        multi = sum(1 for o in ORDERS.values() if o.is_multi)
        exts = sum(1 for o in ORDERS.values() if o.is_extension)
        edit(c.message, f"<b>📊 Статистика</b>\n\nАккаунтов: {s['total']} | Свободно: {s[RentStatus.FREE]} | "
                        f"Ошибки: {s[RentStatus.ERROR]}\nЗавершено: {done} | Возвратов: {refs}\n"
                        f"Мульти-заказов: {multi} | Авто-продлений: {exts}", _back_kb())
    def open_history(c):
        page = _pid(c)
        items = list(ORDERS.values())[::-1]
        total, per = len(items), 10
        pages = max(1, (total + per - 1) // per)
        page = min(max(1, page), pages)
        sl = items[(page - 1) * per:page * per]
        kb = K()
        for o in sl:
            if o.status == RentStatus.FINISHED:
                icon = '✅'
            elif o.status == RentStatus.REFUND:
                icon = '💰'
            elif o.is_extension:
                icon = '🔄'
            elif o.is_multi:
                icon = '📦'
            else:
                icon = '⏳'
            multi = f" x{len(o.acc_ids)}" if o.is_multi and o.acc_ids else ""
            ext = " (продление)" if o.is_extension else ""
            kb.add(B(f"{icon} #{o.id[:8]}... | {o.buyer}{multi}{ext}", None, _CBT.EMPTY))
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
    def open_debug(c):
        lines = [f"<b>🔍 Отладка</b>\n",
                 f"Steam offset: {SteamGuard._time_offset}s",
                 f"Last sync: {SteamGuard._last_sync}",
                 f"Playwright: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}",
                 f"Auto-extend: {'✅' if SETTINGS.auto_extend else '❌'}",
                 f"Cooldowns: {len(_code_cooldowns)}",
                 f"Processed: {len(_processed_orders)}",
                 f"\n<b>Аккаунты:</b>"]
        for a in ACCOUNTS[:10]:
            lines.append(f"• {a.login}: tag='{a.tag}' status={a.status} hrs={_format_periods(a.allowed_hours)}")
        lines.append("\n<b>Лоты:</b>")
        for lid in list(SETTINGS.lots.keys())[:10]:
            lc = SETTINGS.get_lot(lid)
            if lc:
                cnt = f" x{lc.count}" if lc.count > 1 else ""
                lines.append(f"• {lid}: '{lc.tag}' → {_period_label(lc.hours)}{cnt}")
        lines.append("\n<b>Заказы (5):</b>")
        for oid, o in list(ORDERS.items())[-5:]:
            multi = " [MULTI]" if o.is_multi else ""
            ext = " [EXT]" if o.is_extension else ""
            lines.append(f"• {oid[:10]}... buyer_id={o.buyer_id} status={o.status}{multi}{ext}")
        edit(c.message, "\n".join(lines), _back_kb())
    def get_files(c):
        for f in ("settings.json", "accounts.json", "orders.json"):
            p = _get_path(f)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        bot.send_document(c.message.chat.id, fh)
                except Exception:
                    pass
        answer(c)

    tg.cbq_handler(open_main, lambda c: c.data == CBT.MAIN or c.data.startswith(CBT.SP))
    tg.cbq_handler(open_acc_menu, lambda c: c.data == CBT.ACC_MENU)
    tg.cbq_handler(start_add, lambda c: c.data == CBT.ACC_ADD)
    tg.cbq_handler(open_lots, lambda c: c.data == CBT.LOTS)
    tg.cbq_handler(lot_add, lambda c: c.data == CBT.LOT_ADD)
    tg.cbq_handler(open_reviews, lambda c: c.data == CBT.REVS)
    tg.cbq_handler(rev_add, lambda c: c.data == CBT.REV_ADD)
    tg.cbq_handler(open_notifs, lambda c: c.data == CBT.NOTIFS)
    tg.cbq_handler(open_msgs, lambda c: c.data == CBT.MSGS)
    tg.cbq_handler(open_stats, lambda c: c.data == CBT.STATS)
    tg.cbq_handler(open_debug, lambda c: c.data == CBT.DEBUG)
    tg.cbq_handler(hrs_done, lambda c: c.data == CBT.HRS_DONE)
    for pfx, handler in [
        (CBT.ACC_LIST, open_acc_list), (CBT.ACC_DETAIL, open_acc_detail),
        (CBT.ACC_CODE, acc_code), (CBT.ACC_STOP, acc_stop),
        (CBT.ACC_CHPWD, acc_chpwd), (CBT.ACC_EXTEND_DO, acc_extend_do),
        (CBT.ACC_RESET, acc_reset), (CBT.ACC_EDIT_HOURS, acc_edit_hours),
        (CBT.ACC_TOGGLE_HOUR, acc_toggle_hour), (CBT.ACC_SAVE_HOURS, acc_save_hours),
        (CBT.ACC_MANUAL, acc_manual_start), (CBT.ACC_MANUAL_HOURS, handle_manual_hours),
        (CBT.ACC_DEL, acc_del), (CBT.LOT_TAG, lot_tag),
        (CBT.LOT_HRS, lot_hours), (CBT.LOT_CNT, lot_count), (CBT.LOT_DEL, lot_del),
        (CBT.REV_HRS, rev_hours), (CBT.REV_BON, rev_bonus),
        (CBT.REV_DEL, rev_del), (CBT.MSG_EDIT, msg_edit),
        (CBT.TOGGLE, toggle_setting), (CBT.HIST, open_history),
        (CBT.HRS_TGL, hrs_toggle), (CBT.FILES, get_files),
    ]:
        tg.cbq_handler(handler, lambda c, p=pfx: c.data.startswith(f"{p}:"))
    tg.cbq_handler(acc_extend_menu,
                    lambda c: c.data.startswith(f"{CBT.ACC_EXTEND}:") and c.data.count(":") == 1)
    for state, handler in [
        ("ASR_LOGIN", _h_login), ("ASR_PASS", _h_pass),
        ("ASR_TAG", _h_tag), ("ASR_MAN_BUYER", _h_manual_buyer),
        ("ASR_LOT_ID", _h_lot_id), ("ASR_MSG_EDIT", _h_msg_edit),
    ]:
        tg.msg_handler(handler, func=lambda m, s=state: tg.check_state(m.chat.id, m.from_user.id, s))
    bot.register_message_handler(
        _h_mafile, content_types=['document', 'text'],
        func=lambda m: tg.check_state(m.chat.id, m.from_user.id, "ASR_MAFILE"))
    tg.msg_handler(open_main_cmd, commands=['auto_steam_rent'])
    card.add_telegram_commands(UUID, [
        ("auto_steam_rent", "открыть настройки авто аренды аккаунтов", True),
    ])
    threading.Thread(target=rental_check_loop, args=(card,), daemon=True).start()


BIND_TO_PRE_INIT = [init]
BIND_TO_NEW_ORDER = [process_new_order]
BIND_TO_NEW_MESSAGE = [process_message]
BIND_TO_ORDER_STATUS_CHANGED = [process_order_status_changed]
BIND_TO_DELETE = None
