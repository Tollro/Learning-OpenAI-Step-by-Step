from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

model_output = "人工智能，机器学习，深度学习，自然语言处理，计算机视觉"
parsed_output = output_parser.parse(model_output)
print(parsed_output)  # 输出