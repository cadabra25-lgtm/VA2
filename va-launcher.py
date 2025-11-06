# launcher.py
import os
import sys
import importlib.util

def main():
    # Путь к внешнему скрипту — рядом с .exe
    if getattr(sys, 'frozen', False):
        # Запущен как .exe
        app_dir = os.path.dirname(sys.executable)
    else:
        # Запущен как .py
        app_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(app_dir, "va.py")

    if not os.path.exists(script_path):
        print(f"Ошибка: не найден va.py в {app_dir}")
        sys.exit(1)

    # Динамически загружаем и запускаем
    spec = importlib.util.spec_from_file_location("va", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["va"] = module
    spec.loader.exec_module(module)

    # Предполагаем, что в app_logic.py есть функция run()
    if hasattr(module, 'run'):
        module.run()
    else:
        print("Ошибка: в va.py нет функции run()")

if __name__ == "__main__":
    main()