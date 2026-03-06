import os
import sys
import json
import platform


def run_diagnostics():
    print("=========================================")
    print("      🔍 新电脑环境 & Autogen 诊断脚本      ")
    print("=========================================")

    # 1. 检查系统与 Python 版本
    print("\n[1/4] 💻 系统与 Python 检查")
    print(f"  - 操作系统: {platform.system()} {platform.release()}")
    print(f"  - Python 版本: {sys.version.split()[0]}")

    # 2. 检查核心依赖包
    print("\n[2/4] 📦 依赖包检查")
    packages = ["pandas", "numpy", "requests", "autogen"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} 已安装")
        except ImportError:
            print(f"  ❌ 缺失 {pkg}! 请运行: pip install {pkg}")

    # 3. 检查环境变量 (API Keys)
    print("\n[3/4] 🔑 环境变量检查")
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        print("  ✅ POLYGON_API_KEY 已配置 (长度: {})".format(len(polygon_key)))
    else:
        print("  ❌ 缺失 POLYGON_API_KEY! 你的新电脑上没有配置这个环境变量，会导致 Polygon 接口请求失败。")

    # 4. 检查文件路径与配置文件
    print("\n[4/4] 📁 文件路径与 Autogen 配置检查")

    # 这里使用的是你原代码里的路径
    config_path = r"C:\Users\bangc\one-person-unicorn-infra\config_api_keys"
    out_dir = r"C:\Users\bangc\one-person-unicorn-infra\academic_results"

    if os.path.exists(config_path):
        print(f"  ✅ 配置文件路径存在: {config_path}")
        if os.path.isdir(config_path):
            print("  ⚠️ 警告: config_path 是一个【文件夹】！")
            print(
                "     autogen.config_list_from_json() 通常需要指向一个具体的 JSON 【文件】 (例如 OAI_CONFIG_LIST.json)。")
            print("     这就是为什么会导致 'Model config not found' 错误！")
        else:
            # 如果是文件，尝试解析它是否包含 grok-4
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print("  ✅ 配置文件是有效的 JSON 格式")

                # 提取配置中的 model 列表
                models = [item.get("model") for item in data if isinstance(item, dict)]
                if "grok-4" in models:
                    print("  ✅ 在配置文件中找到了 'grok-4' 的配置项")
                else:
                    print(f"  ❌ 配置文件中没有找到 'grok-4'! 目前拥有的模型配置为: {models}")
            except Exception as e:
                print(f"  ❌ 无法读取或解析配置文件: {e}")
    else:
        print(f"  ❌ 配置文件路径不存在: {config_path}")
        print("     提示: 检查新电脑的用户名是否还是 'bangc'，或者代码文件夹是否在这个位置。")

    if os.path.exists(out_dir):
        print(f"  ✅ 输出目录存在: {out_dir}")
    else:
        print(f"  ⚠️ 输出目录不存在: {out_dir} (如果你的代码里有 os.makedirs，这一步可以忽略)")

    print("\n=========================================")
    print("             诊断结束               ")
    print("=========================================")


if __name__ == "__main__":
    run_diagnostics()