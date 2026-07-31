from pathlib import Path
import unittest


class WindowsLauncherTests(unittest.TestCase):
    def test_windows_launcher_has_safe_startup_contract(self):
        script = Path("start_web.bat").read_text(encoding="utf-8")

        self.assertIn('set "HOST=127.0.0.1"', script)
        self.assertIn('set "PORT=8742"', script)
        self.assertIn("py -3", script)
        self.assertIn("sys.version_info >= (3, 9)", script)
        self.assertIn("-m venv", script)
        self.assertIn('start "" "%URL%"', script)
        self.assertIn("winget install Gyan.FFmpeg", script)
        self.assertNotIn("taskkill", script.lower())

    def test_windows_launcher_persists_the_pip_source_choice(self):
        script = Path("start_web.bat").read_text(encoding="utf-8")

        self.assertIn('.pip-source-choice', script)
        self.assertIn('choice /c YN', script)
        self.assertIn('https://pypi.tuna.tsinghua.edu.cn/simple', script)
        self.assertLess(
            script.index("call :configure_pip_source"),
            script.index('if not exist "%VENV_DIR%\\.requirements-installed"'),
        )
