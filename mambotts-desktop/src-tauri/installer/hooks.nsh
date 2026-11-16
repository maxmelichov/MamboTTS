; Releases before BlueV3 (1.3.x) installed eSpeak's phoneme data beside the
; executable. An upgrade only replaces the files it ships, so that directory
; would stay behind for good, and the uninstaller, which removes only what
; this release installed, would leave it too.
!macro NSIS_HOOK_POSTINSTALL
  RMDir /r "$INSTDIR\espeak-ng-data"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  RMDir /r "$INSTDIR\espeak-ng-data"
!macroend
