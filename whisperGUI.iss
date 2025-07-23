[Setup]
AppName=Whisper Transcriber
AppVersion=2.7
DefaultDirName={pf}\WhisperTranscriber
DefaultGroupName=Whisper Transcriber
OutputDir=output
OutputBaseFilename=WhisperTranscriberSetup
Compression=lzma
SolidCompression=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 主程序文件
Source: "dist\whisperGUI.exe"; DestDir: "{app}"; Flags: ignoreversion
; FFmpeg二进制文件
Source: "ffmpeg\bin\ffmpeg.exe"; DestDir: "{app}\ffmpeg\bin"; Flags: ignoreversion
Source: "ffmpeg\bin\ffplay.exe"; DestDir: "{app}\ffmpeg\bin"; Flags: ignoreversion
Source: "ffmpeg\bin\ffprobe.exe"; DestDir: "{app}\ffmpeg\bin"; Flags: ignoreversion
Source: "ffmpeg\bin\*.dll"; DestDir: "{app}\ffmpeg\bin"; Flags: ignoreversion
; ffmpeg环境变量加载

; 模型文件
Source: "models\*"; DestDir: "{app}\models"; Flags: recursesubdirs

[Icons]
Name: "{group}\Whisper Transcriber"; Filename: "{app}\whisperGUI.exe"
Name: "{commondesktop}\Whisper Transcriber"; Filename: "{app}\whisperGUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\whisperGUI.exe"; Description: "Launch Whisper Transcriber"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\ffmpeg"

[Code]
const
  EnvironmentKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

procedure EnvAddPath(Path: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths) then
  begin
    RegWriteStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Path);
  end
  else
  begin
    if Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';') = 0 then
    begin
      Paths := Paths + ';' + Path;
      RegWriteStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths);
    end;
  end;
end;

procedure EnvRemovePath(Path: string);
var
  Paths: string;
  P: Integer;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths) then
  begin
    P := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';');
    if P <> 0 then
    begin
      Delete(Paths, P, Length(Path) + 1);
      RegWriteStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    EnvAddPath(ExpandConstant('{app}\ffmpeg\bin'));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    EnvRemovePath(ExpandConstant('{app}\ffmpeg\bin'));
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
end;
