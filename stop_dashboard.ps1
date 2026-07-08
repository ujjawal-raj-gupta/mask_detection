# Stop dashboard processes that may lock COM4 or the webcam.
Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
	Where-Object { $_.CommandLine -like '*Face-Mask-Detection*dashboard_app*' } |
	ForEach-Object {
		Write-Host "Stopping dashboard (PID $($_.ProcessId))..."
		Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
	}

Write-Host "Done. You can upload to COM4 in Arduino IDE now."
