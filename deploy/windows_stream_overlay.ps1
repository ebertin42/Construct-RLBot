# Always-on-top overlay showing which run the RLViser stream is displaying.
# Reads current_stream.txt (written by scripts/watch_loop.sh over \\wsl$) every
# 2 s. Run alongside windows_viser_relay.ps1 + rlviser.exe:
#   powershell -ExecutionPolicy Bypass -File windows_stream_overlay.ps1
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$statusPath = Join-Path $env:LOCALAPPDATA 'Construct\current_stream.txt'

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Construct stream'
$form.TopMost = $true
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
$form.BackColor = [System.Drawing.Color]::Black
$form.Opacity = 0.78
$form.ShowInTaskbar = $false
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
$form.Size = New-Object System.Drawing.Size(460, 40)
$form.Location = New-Object System.Drawing.Point(24, 24)

$label = New-Object System.Windows.Forms.Label
$label.Dock = [System.Windows.Forms.DockStyle]::Fill
$label.ForeColor = [System.Drawing.Color]::Lime
$label.Font = New-Object System.Drawing.Font('Consolas', 13, [System.Drawing.FontStyle]::Bold)
$label.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
$label.Text = ' waiting for stream...'
$form.Controls.Add($label)

# drag anywhere to move
$dragging = $false; $dragOff = $null
$label.Add_MouseDown({ $script:dragging = $true; $script:dragOff = $_.Location })
$label.Add_MouseMove({ if ($script:dragging) {
    $p = [System.Windows.Forms.Control]::MousePosition
    $form.Location = New-Object System.Drawing.Point(($p.X - $dragOff.X), ($p.Y - $dragOff.Y)) } })
$label.Add_MouseUp({ $script:dragging = $false })
# double-click to close
$label.Add_DoubleClick({ $form.Close() })

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 2000
$timer.Add_Tick({
    if (Test-Path $statusPath) {
        $txt = (Get-Content $statusPath -Raw -ErrorAction SilentlyContinue)
        if ($txt) { $label.Text = ' ' + $txt.Trim() }
    }
})
$timer.Start()
[System.Windows.Forms.Application]::Run($form)
