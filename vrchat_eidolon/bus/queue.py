"""Event queue with backpressure strategy.

In the final design, this will implement policies such as:
- keep only the latest VisionTick
- prioritize UserUtterance over vision ticks
"""
