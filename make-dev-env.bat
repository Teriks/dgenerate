@echo off

python make_dev_env.py
if %errorlevel% neq 0 (
    echo.
    echo Setup failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!