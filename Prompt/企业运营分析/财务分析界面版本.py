import sys
import json
import html  # 新增：用于转义 HTML 特殊字符
import os
from openai import AzureOpenAI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

# ------------------- 配置区域 -------------------
AZURE_ENDPOINT = os.getenv("AZURE_GPT4O_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION = "2025-01-01-preview"

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    print("警告：请先设置环境变量 AZURE_GPT4O_ENDPOINT 和 AZURE_GPT4O_API_KEY")
# --------------------------------------------------

class OpenAICallWorker(QThread):
    finished = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(self, messages, deployment_name, endpoint, api_key, api_version):
        super().__init__()
        self.messages = messages
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version

    def run(self):
        try:
            client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                temperature=0.7,
                max_tokens=2000
            )
            assistant_message = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
            self.finished.emit(assistant_message, total_tokens)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("企业财务分析助手 (Azure OpenAI)")
        self.resize(900, 700)

        self.messages = [
            {"role": "system", "content": "你是一位专业的财务分析师。"}
        ]
        self.current_file_path = ""
        self.is_processing = False

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 文件选择区域
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_btn = QPushButton("选择 JSON 文件")
        self.file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.file_btn)
        main_layout.addLayout(file_layout)

        # 分析按钮
        self.analyze_btn = QPushButton("开始财务分析")
        self.analyze_btn.clicked.connect(self.analyze_financial_data)
        main_layout.addWidget(self.analyze_btn)

        # 对话显示区域
        self.display_area = QTextEdit()
        self.display_area.setReadOnly(True)
        self.display_area.setFont(QFont("Microsoft YaHei", 10))
        main_layout.addWidget(self.display_area, 1)

        # 用户输入区域
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("在此输入追问...")
        self.input_edit.returnPressed.connect(self.send_user_message)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_user_message)
        input_layout.addWidget(self.input_edit, 1)
        input_layout.addWidget(self.send_btn)
        main_layout.addLayout(input_layout)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    # ---------- 新增：文本转 HTML 方法 ----------
    def text_to_html(self, text):
        """将普通文本转换为适合 HTML 显示的格式：转义特殊字符，并将换行符替换为 <br>"""
        escaped = html.escape(text)          # 转义 <>& 等
        return escaped.replace('\n', '<br>')  # 保留换行
    # -----------------------------------------

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择财务数据文件", "", "JSON 文件 (*.json);;所有文件 (*)"
        )
        if file_path:
            self.current_file_path = file_path
            self.file_label.setText(f"已选择: {file_path}")

    def analyze_financial_data(self):
        if not self.current_file_path:
            QMessageBox.warning(self, "警告", "请先选择一个 JSON 文件！")
            return
        if self.is_processing:
            QMessageBox.information(self, "提示", "正在处理中，请稍候...")
            return

        try:
            with open(self.current_file_path, 'r', encoding='utf-8') as f:
                financial_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "文件读取错误", f"无法读取文件：{e}")
            return

        prompt = self.build_analysis_prompt(financial_data)
        self.messages.append({"role": "user", "content": prompt})

        # 显示用户消息时使用 text_to_html
        self.display_area.append(
            f"<b>👤 用户（分析请求）</b><br>{self.text_to_html(prompt)}<hr>"
        )

        self.call_openai()

    def send_user_message(self):
        user_text = self.input_edit.text().strip()
        if not user_text:
            return
        if self.is_processing:
            QMessageBox.information(self, "提示", "正在处理中，请稍候...")
            return

        self.input_edit.clear()
        self.messages.append({"role": "user", "content": user_text})

        # 显示用户消息时使用 text_to_html
        self.display_area.append(
            f"<b>👤 用户</b><br>{self.text_to_html(user_text)}<hr>"
        )

        self.call_openai()

    def call_openai(self):
        self.set_buttons_enabled(False)
        self.is_processing = True
        self.status_bar.showMessage("正在调用 AI 模型，请稍候...")

        self.worker = OpenAICallWorker(
            self.messages,
            DEPLOYMENT_NAME,
            AZURE_ENDPOINT,
            AZURE_API_KEY,
            API_VERSION
        )
        self.worker.finished.connect(self.on_openai_finished)
        self.worker.error.connect(self.on_openai_error)
        self.worker.start()

    def on_openai_finished(self, assistant_message, total_tokens):
        self.messages.append({"role": "assistant", "content": assistant_message})

        # 显示助手回复时使用 text_to_html
        self.display_area.append(
            f"<b>🤖 助手</b><br>{self.text_to_html(assistant_message)}<hr>"
        )
        self.display_area.verticalScrollBar().setValue(
            self.display_area.verticalScrollBar().maximum()
        )

        self.status_bar.showMessage(f"调用完成，本次消耗 {total_tokens} tokens")
        self.set_buttons_enabled(True)
        self.is_processing = False
        self.worker = None

    def on_openai_error(self, error_msg):
        # 显示错误时也使用 text_to_html
        self.display_area.append(
            f"<font color='red'>⚠️ 调用失败，刚才的消息已被丢弃。错误："
            f"{self.text_to_html(error_msg)}</font><hr>"
        )

        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

        self.status_bar.showMessage("调用失败")
        self.set_buttons_enabled(True)
        self.is_processing = False
        self.worker = None

    def set_buttons_enabled(self, enabled):
        self.analyze_btn.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.file_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)

    def build_analysis_prompt(self, financial_data):
        prompt = f"""
你是一位专业的财务分析师。请根据以下公司的财务报表数据，分析该企业的运行状况。重点关注：

1. 盈利能力（毛利率、净利率、ROE等）
2. 偿债能力（流动比率、速动比率、资产负债率）
3. 营运能力（存货周转率、应收账款周转率）
4. 现金流状况（经营现金流净额、自由现金流）
5. 整体评价与风险提示

财务报表数据（JSON 格式）：
{json.dumps(financial_data, indent=2, ensure_ascii=False)}

请提供清晰、专业的分析报告。
"""
        return prompt


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())