Require git
Require python
Require uv

Output docs/assets/demo-cli.gif

Set Width 1400
Set Height 900
Set FontSize 18
Set Framerate 30
Set TypingSpeed 20ms

Hide
Type "python examples/prepare_demo.py --dest data/demo-workspace"
Enter
Wait+Line@30s /Prepared clean demo workspace:/
Ctrl+L
Show

Type 'uv run repopilot run --task-type fix --repo data/demo-workspace "divide() returns the wrong sign; fix it so the tests pass"'
Enter

Wait+Line@300s /Approve this high-risk action/
Type "y"
Enter

Wait+Line@300s /Approve this high-risk action/
Type "y"
Enter

Wait+Line@300s /Approve this high-risk action/
Type "y"
Enter

Wait+Line@300s /Approve this high-risk action/
Type "y"
Enter

Wait+Screen@300s /status=DONE/
Sleep 3s
