"""
Internationalization (i18n) configuration for Singapore Housing Assistant.

Supports English and Chinese languages for UI and AI responses.
"""

from typing import Literal

Language = Literal["en", "zh"]

# =============================================================================
# UI Text Translations
# =============================================================================

UI_TEXT = {
    "en": {
        "title": "# Singapore Housing Rental Assistant",
        "input_placeholder": "Type your question here...",
        "send_button": "Send",
        "clear_button": "Clear Chat",
        "language_label": "Language",
        "chat_label": "Chat",
    },
    "zh": {
        "title": "# 新加坡租房助手",
        "input_placeholder": "请输入您的问题...",
        "send_button": "发送",
        "clear_button": "清空对话",
        "language_label": "语言",
        "chat_label": "对话",
    }
}

# =============================================================================
# Welcome Messages
# =============================================================================

WELCOME_MESSAGES = {
    "en": """**Welcome to Singapore Housing Assistant!**

I'm here to help international students find suitable rental housing in Singapore.

**I can help you with:**
- Understanding HDB vs Condo differences
- Rental price ranges by area
- Housing options near NUS, NTU, SMU
- Transport accessibility and commute times
- Rental process and important tips
- Roommate considerations and room types

**Just ask me anything about renting in Singapore!**

*Example questions:*
- "What's the difference between HDB and Condo?"
- "How much does it cost to rent near NUS?"
- "What areas are good for students on a budget?"
""",
    "zh": """**欢迎使用新加坡租房助手！**

我可以帮助留学生在新加坡找到合适的租房。

**我可以帮您解答：**
- HDB组屋和公寓（Condo）的区别
- 各区域的租金价格范围
- NUS、NTU、SMU 附近的住房选择
- 交通便利性和通勤时间
- 租房流程和注意事项
- 合租须知和房型介绍

**关于新加坡租房，随时问我！**

*示例问题：*
- "HDB和Condo有什么区别？"
- "NUS附近租房要多少钱？"
- "哪些区域适合预算有限的学生？"
"""
}

# =============================================================================
# Language Instruction for LLM
# =============================================================================

LANGUAGE_INSTRUCTIONS = {
    "en": "Please respond to the user in English.",
    "zh": "请用中文回复用户。"
}


def get_ui_text(lang: Language) -> dict:
    """Get UI text for the specified language."""
    return UI_TEXT.get(lang, UI_TEXT["en"])


def get_welcome_message(lang: Language) -> str:
    """Get welcome message for the specified language."""
    return WELCOME_MESSAGES.get(lang, WELCOME_MESSAGES["en"])


def get_language_instruction(lang: Language) -> str:
    """Get LLM language instruction for the specified language."""
    return LANGUAGE_INSTRUCTIONS.get(lang, LANGUAGE_INSTRUCTIONS["en"])
