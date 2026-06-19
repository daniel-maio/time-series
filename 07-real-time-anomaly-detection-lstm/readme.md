# cmd

docker compose -f docker-compose.yml up -d

# Powershell

==========
$client = New-Object System.Net.Sockets.TcpClient("localhost", 9999)
$stream = $client.GetStream()

$data = '{"temperature":[16,17,24,19,20],"humidity":[65,41,42,43,44],"pressure":[1010,1011,1012,1013,1035]}'

$bytes = [System.Text.Encoding]::UTF8.GetBytes($data)
$stream.Write($bytes, 0, $bytes.Length)

$reader = New-Object System.IO.StreamReader($stream)
while ($true)
{$line = $reader.ReadLine()
    if ($null -eq $line)
    {
        break
    }
    Write-Host $line}

=========
# Container

bash

python -m scripts.train

python -m scripts.deploy

Open external terminal

(
  echo '{"temperature":[22.5, 23.0, 180.5, 24.0, 23.0],"humidity":[45.0, 50.0, 98.0, 150.0, 50.0],"pressure":[1013.00, 1012.5, 1013.2, 1013.2, 1.5]}'
) | nc localhost 9999

# Spark Master
http://localhost:9090

# History Server
http://localhost:18080

# cmd
docker compose -f docker-compose.yml down