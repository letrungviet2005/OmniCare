# OmniCare AI

OmniCare AI is an AI-powered elderly care and health monitoring platform designed to support continuous, non-intrusive monitoring of elderly people in home and care environments.

The system combines Computer Vision, Speech Processing, Multimodal AI, Edge AI, and intelligent event processing to detect abnormal situations, analyze daily activities, generate health summaries, and provide an AI Companion for elderly users.

---

## Overview

Elderly people may experience unexpected incidents such as falls, prolonged inactivity, or emergency situations while family members and caregivers are not continuously present.

OmniCare AI aims to address this problem by transforming raw visual and audio data into meaningful health and activity events.

The system is designed around three main capabilities:

- Real-time detection of abnormal events
- Continuous activity and event tracking
- Intelligent interaction and daily health summarization

Instead of simply recording video or audio, OmniCare AI analyzes these inputs and converts them into structured events that can be used for monitoring, alerts, reports, and AI-assisted interaction.

---

## System Architecture

```text
                    Video / Audio Input
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
      Fall Detection   Voice Detection   Conversation
            |               |               |
            +---------------+---------------+
                            |
                            v
                      Event Engine
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        Event Log       Risk Analysis   Alert System
             |
             v
       Daily Summary
             |
             v
       AI Companion
             |
             v
        Web Platform