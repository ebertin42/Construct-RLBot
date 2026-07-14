# UDP relay so RLViser on Windows can receive a Construct stream from WSL2 (NAT mode).
#
# Why this exists: rlviser sends a startup packet to 127.0.0.1:34254; if nothing is
# bound there, Windows poisons its socket with WSAECONNRESET and rlviser goes
# permanently deaf. This relay (a) occupies 127.0.0.1:34254 so that never happens,
# (b) forwards the WSL stream to rlviser from a localhost source, and (c) forwards
# rlviser's replies back to WSL.
#
# Usage (start THIS first, then rlviser.exe, then stream from WSL):
#   powershell -ExecutionPolicy Bypass -File windows_viser_relay.ps1
# WSL side: CONSTRUCT_VISER_ADDR=<windows-host-ip>:45250 python scripts/watch.py <ckpt>

$ErrorActionPreference = 'Stop'

# Socket A: rlviser-facing, source-pinned to 127.0.0.1:34254
$a = New-Object System.Net.Sockets.UdpClient(
    (New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Loopback), 34254))
# Ignore ICMP-unreachable poisoning on our own sockets (SIO_UDP_CONNRESET off)
$SIO_UDP_CONNRESET = -1744830452
[void]$a.Client.IOControl($SIO_UDP_CONNRESET, [byte[]]@(0), $null)

# Socket B: WSL-facing
$b = New-Object System.Net.Sockets.UdpClient(
    (New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any), 45250))
[void]$b.Client.IOControl($SIO_UDP_CONNRESET, [byte[]]@(0), $null)

$rlviser = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Loopback), 45243
$anyEp = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any), 0
$wslSource = $null
$fwd = 0; $back = 0

Write-Host "relay up: WSL -> 0.0.0.0:45250 -> 127.0.0.1:45243 (rlviser); replies return. Ctrl+C to stop."

while ($true) {
    $idle = $true
    while ($b.Available -gt 0) {
        $data = $b.Receive([ref]$anyEp)
        $wslSource = New-Object System.Net.IPEndPoint $anyEp.Address, $anyEp.Port
        [void]$a.Send($data, $data.Length, $rlviser)
        $fwd++; $idle = $false
    }
    while ($a.Available -gt 0) {
        $data = $a.Receive([ref]$anyEp)
        if ($null -ne $wslSource) { [void]$b.Send($data, $data.Length, $wslSource); $back++ }
        $idle = $false
    }
    if ($fwd -gt 0 -and ($fwd % 1000) -eq 0) { Write-Host "forwarded $fwd, returned $back" }
    if ($idle) { Start-Sleep -Milliseconds 2 }
}
