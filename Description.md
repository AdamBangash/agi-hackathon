# Demo Description

This is a Unitree G1 humanoid with two dexterous Inspire hands and a depth camera in its head. Everything you see runs locally, with no cloud anywhere in the loop.

The robot's brain is Gemma 4, running on the robot's own Jetson through Google's LiteRT-LM, with its language model on the onboard GPU. Gemma watches the camera feed, reasons about the scene, and **outputs a running description log of what's happening** — who's in front of the robot and what they're doing — in plain language, from the robot's own hardware.

Alongside Gemma, LiteRT vision models track the person's arms and every finger in real time. We map those to the robot's shoulders, elbows and twelve finger joints, through a motion layer we built for safety: it filters out tracking noise, caps speed and acceleration, holds the torso steady, and stops the arms if the body starts to tip.

Why hands? Hands are where robots meet the real world. Dexterous hands let one robot open doors, use tools and handle fragile objects instead of needing a custom gripper per task. Mirroring a person's hands is the fastest way to teach that: every session is a demonstration that can train a robot to do the task on its own. The same setup lets a person work remotely in hazardous places, helps in care settings, and lets a robot understand and respond to gestures.

A humanoid that sees, describes and moves its hands like a person — all on the device in front of you.
