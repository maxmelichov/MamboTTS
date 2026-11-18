; Files earlier releases installed beside the executable that this one no
; longer ships: eSpeak's phoneme data (1.3.x) and the DirectX Shader Compiler
; for the GPU engine (1.4.0 and 1.4.1; Windows now runs ONNX Runtime). An
; upgrade only replaces the files it ships, so these would stay behind for
; good, and the uninstaller, which removes only what this release installed,
; would leave them too.
!macro MAMBOTTS_REMOVE_RETIRED_FILES
  RMDir /r "$INSTDIR\espeak-ng-data"
  Delete "$INSTDIR\dxcompiler.dll"
  Delete "$INSTDIR\dxil.dll"
  Delete "$INSTDIR\dxc-LICENSE.txt"
!macroend

!macro NSIS_HOOK_POSTINSTALL
  !insertmacro MAMBOTTS_REMOVE_RETIRED_FILES
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  !insertmacro MAMBOTTS_REMOVE_RETIRED_FILES
!macroend
