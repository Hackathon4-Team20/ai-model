import requests
import json
import re
import logging
from typing import Optional, Dict


def default_filter(message: str) -> bool:
    """
    يتحقق مما إذا كانت الرسالة صغيرة جداً أو تحية أو شكراً لتجاهلها.
    """
    greetings = [r"^مرحبا", r"^أهلاً", r"^السلام عليكم", r"^شكراً", r"^thanks"]
    if len(message.strip()) < 5:
        return True
    for pattern in greetings:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False


class SatisfactionTracker:

    def __init__(
        self,
        openrouter_api_key: str,
        initial_score: int = 3,
        model: str = "mistralai/devstral-small:free",
        temperature: float = 0.0,
        summary_interval: int = 5,
    ):
        # إعداد logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.api_key = openrouter_api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        self.score = initial_score
        self.history = []
        self.summary: str = ""
        self.model = model
        self.temperature = temperature
        self.summary_interval = summary_interval
        self._calls = 0

    def add_message(self, role: str, message: str) -> Dict:
        """
        إضافة رسالة جديدة للنظام وتقييمها.
        """
        if role == 'user' and default_filter(message):
            self.logger.debug("تجاهل رسالة فلترة: %s", message)
            # لا نغير الدرجة، نعيدها كما هي
            return {"updated_score": self.score, "status": self._status(), "reason": "رسالة غير دالة"}

        self.history.append({"role": role, "content": message})
        self._calls += 1

        # تحديث الملخص دوريًا بعد عدد معين من الرسائل
        if len(self.history) % self.summary_interval == 0:
            self._update_summary()

        prompt = self._build_prompt(message)

        try:
            response = self._call_llm(prompt)
        except Exception as e:
            self.logger.error("خطأ في الاتصال بالنموذج: %s", e)
            return {"updated_score": self.score, "status": self._status(), "reason": "فشل في الاتصال بالنموذج"}

        result = self._parse_response(response)
        self.logger.info(
            "استدعاء #%d | من:%s | رسالة:'%s' => %s",
            self._calls, role, message, result
        )

        self.score = result["updated_score"]
        return result

    def _build_prompt(self, new_message: str) -> Dict:
        """
        بناء برومبت للنموذج مع تضمين ملخص الحوارات السابقة وآخر 6 رسائل.
        """
        parts = []
        if self.summary:
            parts.append(f"ملخص الحوارات السابقة: {self.summary}")

        for msg in self.history[-6:]:
            prefix = "العميل" if msg['role'] == 'user' else "الدعم"
            parts.append(f"{prefix}: {msg['content']}")

        parts.append(f"الرسالة الجديدة (العميل): {new_message}")
        conversation = "\n".join(parts)

        system_content = (
            "أنت مساعد خبير بتقييم رضا العملاء.\n"
            "قاعدة التقييم: 1 = غير راضٍ تماماً، 5 = راضٍ تماماً.\n"
            "تفسير الرموز التعبيرية (الإيموجي) جزء مهم من تقييم الرضا.\n"
            "مثلاً، 😊 تعني رضا، و😞 تعني عدم رضا.\n"
            "تجاهل التحيات والعبارات غير الدالة.\n"
            "إذا كانت الرسالة لا تحتوي على أي مؤشر لرضا أو عدم رضا (مثل رقم هاتف، تحية، سؤال عام)، أرجع \"updated_score\": 0.\n"
            "أجب دائماً بصيغة JSON: {\n"
            "  \"updated_score\": <int>,\n"
            "  \"status\": \"راضٍ\"|\"غير راضٍ\",\n"
            "  \"reason\": <string>\n"
            "}\n"
        )

        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "system", "content": conversation},
            ]
        }

    def _call_llm(self, payload: Dict) -> str:
        """
        استدعاء API للنموذج والحصول على الرد.
        """
        resp = self.session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']

    def _parse_response(self, text: str) -> Dict:
        """
        تحويل النص المردود من النموذج إلى قاموس Python.
        """
        try:
            result = json.loads(text)
            score = int(result.get('updated_score', self.score))

            # في حالة returned_score == 0، لا نغير الدرجة
            if score == 0:
                score = self.score

            status = result.get('status', self._status(score))
            reason = result.get('reason', '').strip()
            return {"updated_score": score, "status": status, "reason": reason}
        except Exception:
            # محاولة استخراج رقم من النص كدرجة بديلة
            score = self.score
            m = re.search(r"(\d)", text)
            if m:
                score = int(m.group(1))
            status = "راضٍ" if score >= 3 else "غير راضٍ"
            reason = "غير متوفر"
            return {"updated_score": score, "status": status, "reason": reason}

    def _status(self, score: Optional[int] = None) -> str:
        """
        تحديد حالة الرضا بناءً على الدرجة.
        """
        s = score if score is not None else self.score
        return "راضٍ" if s >= 3 else "غير راضٍ"

    def _update_summary(self):
        """
        تحديث الملخص للحفاظ على أداء أفضل مع المحادثات الطويلة.
        """
        segment = self.history[:-6]
        if not segment:
            return
        text_block = " ".join([m['content'] for m in segment])[:2000]

        prompt = (
            "أنت مساعد خبير. لخّص المحادثة التالية في جملة أو جملتين: '" + text_block + "'"
        )
        try:
            summary_resp = self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": prompt}
                    ]
                }
            )
            summary_resp.raise_for_status()
            summary_text = summary_resp.json()['choices'][0]['message']['content']
            self.summary = summary_text.strip()
            self.history = self.history[-6:]
            self.logger.debug("تم تحديث الملخص: %s", self.summary)
        except Exception as e:
            self.logger.warning("فشل تحديث الملخص: %s", e)


# ======= مثال على الاستخدام =======
# tracker = SatisfactionTracker(openrouter_api_key="API_KEY_HERE")
# print(tracker.add_message('user', 'الخدمة سيئة جدا'))
# print(tracker.add_message('assistant', 'نعتذر عن الخطأ'))
# print(tracker.add_message('user', 'الآن بدأت تتحسن الأمور'))
