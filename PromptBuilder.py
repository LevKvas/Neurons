class PromptTemplates:
    @staticmethod
    def build_messages(text: str, examples: dict) -> list:
        # form examples
        few_shot_context = ""
        for cat, ex_text in examples.items():
            few_shot_context += f"Текст: {ex_text[:200]}...\nКатегория: {cat}\n\n"

        # allowed categories
        allowed = ", ".join(examples.keys())

        system_content = (
            "Вы — узкоспециализированный классификатор. "
            "Ваша задача — называть категорию текста строго из предложенного списка."
        )

        user_content = (
            f"### ПРИМЕРЫ ДЛЯ ОБУЧЕНИЯ:\n\n{few_shot_context}"
            f"### ЗАДАНИЕ:\n"
            f"Классифицируй следующий текст. \n"
            f"Текст: {text}\n\n"
            f"ВНИМАНИЕ: Выбери только одно название из списка: [{allowed}].\n"
            f"Ответ (только одно слово):"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]