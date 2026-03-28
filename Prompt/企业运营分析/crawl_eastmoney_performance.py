import requests
import json
import re
from datetime import datetime

def fetch_performance_data(stock_code="688027", cookie=None):
    """
    从东方财富API获取业绩报表数据
    
    Args:
        stock_code: 股票代码，默认为688027
        cookie: 浏览器Cookie字符串（可选，如果不提供则尝试无Cookie访问）
    
    Returns:
        dict: 包含业绩报表数据的字典
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    
    params = {
        "sortColumns": "REPORTDATE",
        "sortTypes": "-1",
        "pageSize": "50",
        "pageNumber": "1",
        "columns": "ALL",
        "filter": f"(SECURITY_CODE=\"{stock_code}\")",
        "reportName": "RPT_LICO_FN_CPD"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://data.eastmoney.com/bbsj/yjbb/{stock_code}.html",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    
    # 如果提供了Cookie，添加到请求头
    if cookie:
        headers["Cookie"] = cookie
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        response.raise_for_status()
        
        # 处理JSONP响应
        text = response.text
        if text.startswith('jQuery'):
            json_match = re.search(r'\(({.*})\);?$', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                raise ValueError("无法解析JSONP响应")
        else:
            data = response.json()
        
        if data.get('success') and data.get('result'):
            result = data['result']
            records = result.get('data', [])
            
            cleaned_records = [clean_record(record) for record in records]
            
            return {
                "stock_code": stock_code,
                "stock_name": records[0].get('SECURITY_NAME_ABBR', '') if records else '',
                "total_count": result.get('count', 0),
                "data": cleaned_records,
                "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            print(f"API返回错误: {data.get('message', '未知错误')}")
            return None
            
    except requests.RequestException as e:
        print(f"网络请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"处理失败: {e}")
        return None

def clean_record(record):
    """清理单条数据记录"""
    field_mapping = {
        "REPORTDATE": "报告期",
        "BASIC_EPS": "每股收益(元)",
        "DEDUCT_BASIC_EPS": "每股收益扣除(元)",
        "TOTAL_OPERATE_INCOME": "营业总收入(元)",
        "PARENT_NETPROFIT": "净利润(元)",
        "WEIGHTAVG_ROE": "净资产收益率(%)",
        "YSTZ": "营收同比增长(%)",
        "SJLTZ": "净利润同比增长(%)",
        "BPS": "每股净资产(元)",
        "MGJYXJJE": "每股经营现金流(元)",
        "XSMLL": "销售毛利率(%)",
        "ASSIGNDSCRPT": "利润分配",
        "ZXGXL": "股息率(%)",
        "NOTICE_DATE": "首次公告日期",
        "UPDATE_DATE": "最新公告日期",
        "DATATYPE": "报告类型",
        "SECURITY_NAME_ABBR": "股票名称",
        "TRADE_MARKET": "交易市场"
    }
    
    cleaned = {}
    for key, value in record.items():
        cn_key = field_mapping.get(key, key)
        if value is None or value == "null":
            cleaned[cn_key] = None
        elif key in ["REPORTDATE", "NOTICE_DATE", "UPDATE_DATE"] and value:
            cleaned[cn_key] = str(value)[:10] if value else None
        else:
            cleaned[cn_key] = value
    return cleaned

def save_to_json(data, filename=None):
    if filename is None:
        stock_code = data.get('stock_code', 'unknown')
        filename = f"{stock_code}_performance_report.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"数据已保存到: {filename}")
    return filename

def print_preview(data, limit=3):
    if not data or not data.get('data'):
        print("无数据可预览")
        return
    print(f"\n股票代码: {data['stock_code']}")
    print(f"股票名称: {data['stock_name']}")
    print(f"数据条数: {data['total_count']}")
    print(f"抓取时间: {data['fetch_time']}")
    print("\n--- 数据预览 (最近3期) ---\n")
    for i, record in enumerate(data['data'][:limit]):
        print(f"第{i+1}期: {record.get('报告期', '未知')}")
        print(f"  每股收益: {record.get('每股收益(元)', 'N/A')}")
        income = record.get('营业总收入(元)')
        if income:
            print(f"  营业总收入: {income/1e8:.2f}亿" if income >= 1e8 else f"  营业总收入: {income/1e4:.2f}万")
        profit = record.get('净利润(元)')
        if profit:
            print(f"  净利润: {profit/1e8:.2f}亿" if profit >= 1e8 else f"  净利润: {profit/1e4:.2f}万")
        print(f"  净资产收益率: {record.get('净资产收益率(%)', 'N/A')}%")
        print()

def main():
    stock_code = "688027"
    
    # 从浏览器复制的Cookie（请替换为您自己的）
    # 获取方法：打开浏览器开发者工具 -> Network -> 找到该API请求 -> 复制Cookie值
    my_cookie = "qgqp_b_id=df3fcd28a36a6bd677cc6011b064ec63; fullscreengg=1; fullscreengg2=1; st_nvi=R1ppY95AD63KgAU-Qt5Dw51d5; st_si=25759648359482; nid18=092f4949411cb0f99c017bc404058a2a; nid18_create_time=1774682457499; gviem=lW210Y3yLzGg_LwrqicdJ11a1; gviem_create_time=1774682457499; rskey=u5iMAaStZNW4rZXZIRjIwYlpjSUthWGc1Zz09KaDe6; st_asi=delete; JSESSIONID=4D796E7C3EC943D121884F5270C7327E; st_pvi=42454379514812; st_sp=2025-05-27%2019%3A59%3A35; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=14; st_psi=20260328161603757-113300301075-1240533542"  # 这里替换为完整的Cookie
    
    print(f"正在抓取股票 {stock_code} 的业绩报表数据...")
    
    # 方法1：使用Cookie（推荐）
    data = fetch_performance_data(stock_code, cookie=my_cookie)

    # print(data)
    
    # 方法2：如果不提供Cookie，可能也能获取到数据（但可能不稳定）
    # data = fetch_performance_data(stock_code)
    
    if data and data.get('data'):
        print(f"成功获取 {len(data['data'])} 条数据记录")
        # print_preview(data)
        save_to_json(data, filename=f"./prompt/企业运营分析/{stock_code}_performance_report.json")
        print(f"\n成功获取{stock_code}的业绩数据！\n数据已保存。")
    else:
        print("未能获取到数据，请检查Cookie是否有效或网络连接")

if __name__ == "__main__":
    main()