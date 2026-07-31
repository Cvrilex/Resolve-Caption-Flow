@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_DIR=%~dp0"
set "HOST=127.0.0.1"
set "PORT=8742"
set "URL=http://%HOST%:%PORT%/"
set "STATUS_URL=%URL%api/system/status"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%PROJECT_DIR%requirements.txt"
set "PIP_SOURCE_FILE=%VENV_DIR%\.pip-source-choice"
set "PIP_INDEX_ARGS="

cd /d "%PROJECT_DIR%" || goto :project_error

echo.
echo ========================================
echo  Resolve Caption Flow WebUI
echo ========================================
echo 项目目录: %CD%
echo 访问地址: %URL%
echo.

set "PYTHON_CMD="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if defined PYTHON_CMD goto :python_found

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if defined PYTHON_CMD goto :python_found

echo 未找到 Python 3.9 或更高版本。
echo 请从 https://www.python.org/downloads/windows/ 安装 Python 3.9 或更高版本，
echo 并在安装时勾选 “Add Python to PATH”，然后重新双击本脚本。
goto :error

:python_found
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if errorlevel 1 goto :prepare_environment

powershell -NoProfile -Command "$root=(Resolve-Path -LiteralPath '%PROJECT_DIR%').Path.TrimEnd('\'); try { $status=Invoke-RestMethod -Uri '%STATUS_URL%' -TimeoutSec 2 -ErrorAction Stop; if ([string]$status.service.app_root -eq $root) { exit 0 }; exit 2 } catch { exit 1 }"
if errorlevel 2 goto :port_other_service
if errorlevel 1 goto :port_unrecognized

echo 当前项目的 WebUI 已在运行，正在打开浏览器...
start "" "%URL%"
goto :end

:port_other_service
echo 端口 %PORT% 已被另一个字幕助手实例占用。
echo 为避免终止不属于本脚本的进程，脚本未做任何关闭操作。
echo 请关闭该实例后重试，或在浏览器访问 %URL%。
goto :error

:port_unrecognized
echo 端口 %PORT% 已被其他程序占用，无法安全地启动 WebUI。
echo 请关闭占用端口的程序后重试。
netstat -ano | findstr /r /c:":%PORT% .*LISTENING"
goto :error

:prepare_environment
if exist "%VENV_PYTHON%" goto :install_dependencies

echo 正在创建本地 Python 环境...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_error

:install_dependencies
call :configure_pip_source
if not exist "%REQ_FILE%" goto :check_dependencies
if not exist "%VENV_DIR%\.requirements-installed" goto :install_requirements
if not exist "%VENV_DIR%\.requirements.txt" goto :install_requirements
fc /b "%REQ_FILE%" "%VENV_DIR%\.requirements.txt" >nul
if errorlevel 1 goto :install_requirements
goto :check_dependencies

:install_requirements
echo 正在安装或更新 Python 依赖；首次启动可能需要几分钟...
"%VENV_PYTHON%" -m pip install %PIP_INDEX_ARGS% --upgrade pip
if errorlevel 1 goto :dependency_error
"%VENV_PYTHON%" -m pip install %PIP_INDEX_ARGS% -r "%REQ_FILE%"
if errorlevel 1 goto :dependency_error
copy /y "%REQ_FILE%" "%VENV_DIR%\.requirements.txt" >nul
type nul > "%VENV_DIR%\.requirements-installed"

:check_dependencies
"%VENV_PYTHON%" -c "import fastapi, multipart, pypdf, requests, uvicorn" >nul 2>&1
if not errorlevel 1 goto :check_ffmpeg

echo 检测到 Python 依赖不完整，正在重新安装...
if not exist "%REQ_FILE%" goto :dependency_error
goto :install_requirements

:check_ffmpeg
echo.
echo 正在检查 FFmpeg / FFprobe...
where ffmpeg >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
where ffprobe >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
echo   FFmpeg 和 FFprobe 已可用。
goto :check_resolve

:ffmpeg_missing
echo   未检测到完整的 FFmpeg 套件。WebUI 仍会启动，但视频探测、音频提取和渲染校验可能不可用。
echo   可任选一种方式安装，然后重新打开本窗口：
echo     winget install Gyan.FFmpeg
echo     choco install ffmpeg
echo     scoop install ffmpeg
echo   或手动下载 FFmpeg，并将其 bin 目录加入系统 PATH。

:check_resolve
echo.
echo 正在检查 DaVinci Resolve...
if exist "%ProgramFiles%\Blackmagic Design\DaVinci Resolve\Resolve.exe" goto :resolve_found
echo   未检测到 DaVinci Resolve。仍可生成 SRT，但不能自动渲染成片。
goto :start_server

:resolve_found
echo   已检测到 DaVinci Resolve。

:start_server
echo.
echo 正在启动 Web 服务...
echo 正在打开浏览器。若页面暂未响应，请等待服务启动完成后刷新。
echo 按 Ctrl+C 可停止服务。
start "" "%URL%"
"%VENV_PYTHON%" app\web_server.py --host "%HOST%" --port "%PORT%"
goto :end

:project_error
echo 无法进入项目目录。
goto :error

:venv_error
echo 创建 Python 虚拟环境失败。请确认当前目录有写入权限，并检查 Python 安装。
goto :error

:dependency_error
echo Python 依赖安装失败。请检查网络连接后重试；也可在项目目录运行：
echo "%VENV_PYTHON%" -m pip install -r "%REQ_FILE%"
goto :error

:error
echo.
pause

:end
endlocal
goto :eof

:configure_pip_source
set "PIP_INDEX_ARGS="
if not exist "%PIP_SOURCE_FILE%" goto :ask_pip_source
set /p "PIP_SOURCE=" < "%PIP_SOURCE_FILE%"
if /i "%PIP_SOURCE%"=="tsinghua" goto :use_tsinghua_pip
if /i "%PIP_SOURCE%"=="official" goto :use_official_pip

:ask_pip_source
echo.
echo 首次安装 Python 依赖：如网络访问 PyPI 较慢或失败，可改用清华 PyPI 镜像。
choice /c YN /n /m "是否改用清华镜像"
if errorlevel 2 goto :use_official_pip

:use_tsinghua_pip
set "PIP_SOURCE=tsinghua"
set "PIP_INDEX_ARGS=--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
> "%PIP_SOURCE_FILE%" echo tsinghua
exit /b

:use_official_pip
set "PIP_SOURCE=official"
> "%PIP_SOURCE_FILE%" echo official
exit /b
