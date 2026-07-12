def to_frontend_dict(result) -> dict:
    return {
        "platform": result.platform_name,
        "final_text": result.final_text,
        "final_char_count": result.final_char_count,
        "mechanism": result.mechanism,
        "attempts": [
            {"attempt": a.attempt_number, "text": a.text, "char_count": a.char_count,
             "passed": a.passed, "mechanism": a.mechanism}
            for a in result.attempts
        ],
    }