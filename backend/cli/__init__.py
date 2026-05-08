"""
CLI 模块 — 与 REST API 共享 Engine 层。

bin/danqing-* → DanQingImageEngine / DanQingVideoEngine → ImagePipeline / VideoPipeline

架构分层：
  REST API: routes → Engine → Pipeline → Runtime/Models
  CLI:      bin/*  → Engine → Pipeline → Runtime/Models
"""
