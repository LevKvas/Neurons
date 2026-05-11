class PromptTemplates:
    @staticmethod
    def build_messages(text: str, examples: dict) -> list:
        # Cut to 300 to keep the meaning
        few_shot_context = ""
        for cat, ex_text in examples.items():
            few_shot_context += f"Текст: {ex_text[:300]}\nКатегория: {cat}\n\n"

        allowed = ", ".join(examples.keys())

        system_content = "Ты — бот-классификатор. Твоя задача: прочитать текст и написать название категории."

        user_content = (
            f"Список категорий: {allowed}\n\n"
            f"Примеры:\n{few_shot_context}"
            f"Задание: Определи категорию для текста ниже.\n"
            f"ВАЖНО: Politics — это власть/законы. Travel — это туризм. \
            Entertainment — это про кино, музыку, концерты и отдых.\n\n"
            f"Текст: {text}\n"
            f"Категория:"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]