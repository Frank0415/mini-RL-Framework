"""后续并发训练中保存已完成的回答组。ExperienceBuffer 管理容量和版本准入。
同步版本由 DataEngine 保存并消费本轮回答组。

Store completed response groups for future concurrent training. ExperienceBuffer manages capacity and version admission.
The synchronous version uses DataEngine to store and consume the current iteration's groups.
"""
