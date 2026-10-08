# TACC AgentForge

TACC AgentForge is a course project for **Advanced AI Computing and Systems**. The project explores how to deploy a large language model on the TACC platform and build an agent system on top of the deployed model.

## Project Goals

- Deploy and serve a large language model on TACC.
- Build an agent that can reason, use tools, and complete multi-step tasks.
- Evaluate the system in terms of correctness, latency, throughput, and resource utilization.
- Study practical systems issues such as model loading, GPU scheduling, communication overhead, and inference optimization.

## Planned Architecture

```text
User
  |
  v
Agent Runtime
  |- Planning and reasoning
  |- Tool selection and execution
  |- Memory and context management
  |
  v
LLM Inference Service
  |
  v
TACC Compute Resources
```

The exact model, inference framework, and agent framework will be selected during implementation and documented here.

## Prerequisites

- A registered TACC account and SSH public key
- The TACC `tcloud` SDK
- Python 3.10 or later
- Access to the compute resources required by the selected model

## TACC Setup

Configure the local `tcloud` client with your TACC username and private key:

```bash
tcloud config --username <TACC_USERNAME>
tcloud config --file <SSH_PRIVATE_KEY_PATH>
tcloud init
```

Do not commit private keys, passwords, access tokens, model credentials, or generated cluster configuration containing sensitive information.

## Planned Repository Structure

```text
tacc-agentforge/
|- agent/          # Agent planning, tools, and memory
|- inference/      # Model loading and inference service
|- configs/        # Model, agent, and experiment configurations
|- scripts/        # TACC deployment and job scripts
|- evaluations/    # Benchmarks and evaluation tasks
|- tests/          # Automated tests
|- docs/           # Design notes and experiment reports
`- README.md
```

Directories will be added as the implementation develops.

## Roadmap

- [x] Initialize the project repository
- [ ] Select the base model and inference framework
- [ ] Run a minimal inference job on TACC
- [ ] Expose the model through a reusable inference interface
- [ ] Implement the initial agent loop
- [ ] Add tools and memory
- [ ] Design evaluation tasks and baselines
- [ ] Measure latency, throughput, and GPU utilization
- [ ] Document results and system limitations

## Status

This project is currently in the design and environment-setup stage.

## Course

**Advanced AI Computing and Systems**

## License

No license has been selected yet. Unless a license is added, all rights are reserved by the repository owner.
