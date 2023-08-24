# Wrapper function for creating Chinese prompts.
def make_prompt_zh(prefix, continuation, eval_type="direct", task="word_pred", options=None):
    full_sentence = prefix + continuation
    
    # For "direct" evaluation, just use the full sentence.
    if eval_type == "direct":
        return full_sentence
    
    # Otherwise, the prompt structure will depend on the task/experiment.
    if task == "word_pred":
        if eval_type == "metaQuestionSimple":
            prompt = f"哪一个词最有可能接在下面这句话之后？\n\n{full_sentence}"
        elif eval_type == "metaInstruct":
            prompt = f"你是一个善于帮助的写作助理。请告诉我哪一个词最有可能接在下面这句话之后：\n\n{full_sentence}"
        elif eval_type == "metaQuestionComplex":
            prompt = f"以下是一句话的开头： {prefix}…… 哪一个词最适合接在其后？\n\n答案：{continuation}"
        else:
            raise NotImplementedError
            
    elif task == "word_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        else:
            assert len(options) == 2
            option_str = f"{options[0]}，还是{options[1]}"
            if eval_type == "metaQuestionSimple":
                prompt = f"哪一个词更有可能接在下面这句话之后（{option_str}）？\n\n{full_sentence}"
            elif eval_type == "metaInstruct":
                prompt = f"你是一个善于帮助的写作助理。请告诉我哪一个词更有可能接在下面这句话之后（{option_str}？）：\n\n{full_sentence}"
            elif eval_type == "metaQuestionComplex":
                prompt = f"以下是一句话的开头：{prefix}…… 哪一个词更有可能接在其后：{option_str}？\n\n答案：{continuation}"
            else:
                raise NotImplementedError
                
    elif task == "sentence_judge":
        # Call a separate function for Experiment 3a (sentence judgment in isolation)
        prompt = _make_prompt_yesno(prefix, continuation, eval_type)
                
    elif task == "sentence_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        # Call a separate function for Experiment 3b (sentence comparison)
        prompt = _make_prompt_forcedchoice(continuation, eval_type, options)
    
    else:
        raise ValueError(f"Unknown task specification! '{task}'")
    
    return prompt

def _make_prompt_forcedchoice(continuation, eval_type, options):
    sentence1, sentence2 = options
    option_str = f"1) {sentence1} 2) {sentence2}"
    response_info = "请用1或者2来回答。"
    if eval_type == "metaQuestionSimple":
        prompt = f"以下哪一句话在中文里更为通顺？{option_str} {response_info}\n\n{continuation}"
    elif eval_type == "metaInstruct":
        prompt = f"你是一个善于帮助的写作助理。请告诉我以下哪一句话在中文里更为通顺？\n\n{option_str} {response_info}\n\n{continuation}"
    elif eval_type == "metaQuestionComplex":
        prompt = f"以下是两句话：\n\n{option_str} 哪一句话在中文里更为通顺？{response_info}\n\n答案：{continuation}"
    else:
        raise NotImplementedError
    return prompt

def _make_prompt_yesno(sentence, continuation, eval_type):
    response_info = "请用“是”或者“否”来回答。"
    if eval_type == "metaQuestionSimple":
        prompt = f"下面这句话在中文里是否通顺？ {sentence} {response_info}\n\n{continuation}"
    elif eval_type == "metaInstruct":
        prompt = f"你是一个善于帮助的写作助理。请告诉我以下这句话在中文里是否通顺。\n\n{sentence} {response_info}\n\n{continuation}"
    elif eval_type == "metaQuestionComplex":
        prompt = f"以下是一句话：\n\n{sentence} 这句话在中文里是否通顺？{response_info}\n\n答案：{continuation}"
    else:
        raise NotImplementedError
    return prompt