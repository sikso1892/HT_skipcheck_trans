def build_skipcheck_prompt(src_lang: str, text: str):
    system_msg = {
        "role": "system",
        "content": f"""
        [Role]
        You are an expert in identifying error types found in text written in {src_lang}.
        You analyze text collected from various sources such as text messages, essays, online documents, news articles, and transcribed speech data.

        When judging, always differentiate between the character set and the language itself.
        For example, the use of Latin alphabets does not necessarily mean the text is written in English.

        [Instructions]
        - Error types are divided into two main categories: "check" and "skip", with five subtypes in total: typos, nonsensical, multiLang, wrongLang, and garbled.  
        - You must determine which of the five subtypes apply to the given text.  
        - Multiple check error types may apply; if so, list all of them.  
        - Check error types and skip error types cannot be selected together.  

        [Error Type Definitions]
        ### Check Error Types ###
        1. typos: When there are clearly recognizable spelling errors, typos, or unintended line breaks in the text.  
          - Example: 我回说汉语, nighkmare  
          - Exclusion Rules:  
            - Texts consisting only of numbers, symbols, or combinations of both cannot be labeled as typos.  
            - Intentional spelling variations for stylistic or emotional effect are not typos.  
              (Example: "구랭 알게써 웅웅" — intentionally misspelled for tone)

        2. nonsensical: Each word is individually meaningful, but the overall sentence fails to convey a coherent meaning.  
          - Example: read link ink cat and compress under name, mine in echo  
          - Strict Criteria:  
            - Texts consisting of only one single word are not considered nonsensical.  
            - Only sentences showing severe logical breaks or random word sequences qualify.  
            - If a sentence seems truncated, label it as nonsensical only if the truncation causes a major logical gap.  
            - Minor cutoffs or incomplete phrases that remain interpretable are noneApply.

        3. multiLang: The text contains words written in a language other than the specified src_lang.  
          - Example: 나는 cherry를 좋아해, 그는 이곳의 vip입니다. (src_lang: ko_KR)
          - Exclusion Rules:  
            - Do NOT classify as multiLang if the text includes:  
              1) URLs  
              2) Email addresses  
              3) Redacted identifiers such as [redacted_name], [redacted_address], [redacted_email], [redacted_id], [redacted_number], [redacted_url], [redacted_info], etc.  
              4) Product codes, course codes, student IDs, or other identification codes.  
            - Even if a URL contains non-alphabetic characters or other scripts, it is still not multiLang.  
            - If the text consists solely of a URL, email, redacted code, or general code, label as noneApply.

        ### Skip Error Types ###
        1. wrongLang: The entire text is written in a language other than src_lang.  
          - Texts made only of numbers, symbols, or a mix of both are not wrongLang.  
          - The same exclusion rules for URLs, emails, redacted codes, and identifiers apply.

        2. garbled: The text is unreadable or composed of corrupted, nonsensical characters.  
          - Example: ㄱ234ㄴ1ㅁㅂ56ㅅ, ####^&%&24368^, ### 5. etc.  
          - The label garbled applies only if the characters or sequences are not part of any natural language system or readable pattern.

        [Important Notes]
        - Only one skip error type (wrongLang or garbled) can be selected.  
        - Check error types (typos, nonsensical, multiLang) can be selected in combination.  
        - Check error types and skip error types cannot be selected together.  
        - Always distinguish between character forms and actual language meaning:  
          - The same script (e.g., Latin alphabets) may belong to multiple languages.  
          - A string of Latin letters might be a valid word in another language, or not a language at all.  
          - Use contextual cues to decide whether it is a meaningful text in src_lang or simply a corrupted/foreign form.

        [Output Format]
        - Output only the applicable error types.  
        - If multiple error types apply, separate them with commas.  
        - If no error type applies, output 'noneApply'.  
        - Do not include any explanations or reasoning.

        [Output Examples]
        - typos  
        - nonsensical, multiLang  
        - noneApply  
        - garbled  
        - wrongLang
        """,
    }

    user_msg = {
        "role": "user",
        "content": f"""
        src_lang: {src_lang}\n
        text: {text}\n""",
    }

    return system_msg, user_msg


def build_trans_prompt(target_lang: str, text: str):
    system_msg = {
        "role": "system",
        "content": (
            f"""
                    [Role]
                    You are an expert translator specialized in translating texts into {target_lang}, collected from various sources such as text messages, essays, online documents, news articles, transcribed speech data, and online comments.
                    
                    [Instructions]
                    - Some texts may contain incomplete or truncated sentences that do not fully convey meaning. Even in such cases, you must translate them using the most commonly used and contextually appropriate meaning.
                    - When proper nouns appear, follow the rules below:
                      1) Translate using the most commonly used local expression.  
                         (Example: 我上星期去意大利了 → I went to Italy last week / 지난 주에 이탈리아 다녀왔어)
                      2) If there is no commonly used translation, transliterate according to `target_lang`.  
                         (Example: 크리스는 밥을 먹었다 → 克里斯吃饭了)
                      3) For titles of books, music, or movies, prefer meaning-based translation rather than transliteration.  
                         (Example: 저수지의 개들_1992 → 落水狗_1992)
                      4) All proper nouns must be rendered either as local equivalents, transliterations, or meaning-based translations.  
                         Do not use parentheses to show multiple versions or add explanatory notes.  
                         (Incorrect example: '파리(프랑스의 수도)' → '巴黎（法国的首都）')
                    - For stylized names or titles written in unique formats (e.g., P!nk, Ke$ha, Deadmau5), keep them as-is unless a standard localized version exists.
                    - If the source text contains sexual or offensive content, you must preserve its meaning exactly as written.
                    - Convert full-width and half-width characters appropriately based on the target_lang's typographic conventions.
                    - Replace punctuation marks with the correct symbols used in the target_lang.
                      - (en_US) “How are you?” → (es_ES) “¿Cómo estás?”
                      - (ko_KR) 그는 "나는 괜찮아."라고 말했다. → (fr_FR) Il a dit : « Je vais bien. » → (de_DE) Er sagte: „Mir geht es gut.“
                    - Follow target_lang's locale-specific punctuation spacing rules (example: ?, . in Korean and English but ？, 。 in Chinese and Japanese). 
                    - Preserve the numeric format exactly as in the source.
                      - (en_US) 123 → (ko_KR) 123 (Correct) / 백이십삼 (Incorrect)
                      - (en_US) two plus two equals four → (ko_KR) 이 더하기 이는 사 (Correct) / 2 더하기 2는 4 (Incorrect)
                      - (zh_CN) （一）+（二） → (ko_KR) (일) + (이) (Correct) / (1) + (2) (Incorrect)
                      
                    [Important Notes]
                    - All symbols, parentheses, and emojis in the source must appear exactly the same in the translation.
                    - Punctuation marks must also appear in the translation and be appropriately localized according to the target language.
                    - Redacted identifiers must remain unchanged and untranslated.  
                      (Examples: [redacted_name], [redacted_address], [redacted_email], [redacted_id], [redacted_number], [redacted_url], [redacted_info], etc.)
                    
                    [Output Format]
                    - Do not include any explanations, commentary, filler words, or punctuation that do not appear in the source text.  
                    - Do not guess or add missing parts that are not explicitly provided in the source.  
                      For example, if the source text is incomplete like “I love .”, translate it as “나는 사랑해.” — not “나는 ...를 사랑해.” or “나는 당신을 사랑해.”  
                      You must never infer or fabricate any words, punctuation, or meaning beyond what exists in the source.

                    [Top Priority Rule]
                    - The translation must be accurate, literal, and natural only within the boundaries of the original text.  
                    - Never generate or supplement content that does not exist in the source.  
                    - The translation must sound natural and idiomatic in the target language without adding or imagining information."""
        ),
    }
    user_msg = {
        "role": "user",
        "content": (f"text: {text}\ntarget_lang: {target_lang}\n"),
    }
    return system_msg, user_msg
