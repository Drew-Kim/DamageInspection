# Damage Inspection Prototype: AGV/AMR-Inspired Damage Inspection System with UNIS  
**CS 179M AI Senior Design Project | Raspberry Pi 5 (8GB) + Raspberry Pi AI Camera**

## Summary
This project develops an **AI vision prototype** to detect visible packaging damage in warehouses. The system runs using a **Raspberry Pi 5 (8GB)** and **Raspberry Pi AI Camera**, with future alignment to AMR/AGV deployment.

## Problem and Objective
Damaged packages increase return costs, disrupt operations, and reduce fulfillment quality.  
Our objective is to deliver a **low-cost, reproducible inspection pipeline** that flags damage early and supports downstream automation.

## Scope
Given external data access limits and a 10-week timeline, we adopted a **computer-vision-first scope**.

### Why this scope
- Third-party production camera feeds are currently inaccessible.
- Full robot integration introduces high control/integration overhead.
- A computer vision deliverable is feasible and testable within constraints.

### Current priority
**Data collection** is the primary blocker and focus, since it directly affects model training quality and edge inference performance.

### Data Collection
- Totalled 3000 - 3500 images in total
- Images are split into labels: undamaged_box, damaged_box, opened_box
- Majority of the images are public data set found online, while the remaining images are gathered from UNIS/created at home
- Due to the 10 week time constraint the requirement of public datasets was necessary to gauruntee the accuracy of correctness of the YOLOV8 model.

## Hardware Usage

### A. Canakit Raspberry Pi 5 (8GB) Starter Kit
- 8GB RAM provides stability for the camera and process workloads.
- Starter kit reduces compatibility risk and setup time, such as power, cooling, and case.

### B. Raspberry Pi AI Camera
- Strong ecosystem compatibility with the Raspberry Pis.
- Faster development and fewer integration failures for performance cost balancing.

## Software Dependencies


## Technical Workflow
1. Collect and consolidate datasets from multiple sources  
2. Sort and label images into classes  
3. Train a lightweight detection model  
4. Export and optimize for Raspberry Pi implementation  
5. Deploy on Pi 5 + AI Camera  
6. Run live detection and log results  
7. Test using error analysis using false positives/negatives

## Milestones
- [x] Scope definition
- [x] Hardware selection and budget justification  
- [x] Initial prototype assembly  
- [X] Dataset expansion and class balancing  
- [X] Baseline model training  
- [X] Performance tuning  
- [X] Final Model/Demo

