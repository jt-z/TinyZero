# 仓库分支作用梳理

本仓库的分支根据其命名和功用，可以主要分为以下几类：

### 1. 主分支 (main)

*   **`main` / `remotes/origin/main`**:
    *   这是项目的主干分支，包含了最稳定和最新的代码。
    *   `origin/HEAD -> origin/main` 表明 `main` 是远程仓库 (`origin`) 的默认分支。所有新的 Pull Request 通常会合并到这个分支。

### 2. 功能开发分支 (Feature Branches)

这些分支用于开发新功能或进行实验，以 `wip` (Work In Progress) 结尾或以开发者标识作为前缀是常见做法。

*   **`remotes/origin/countdown-wip`**:
    *   一个正在开发中的、与 "countdown" 相关的功能分支。
*   **`remotes/origin/len_reward`**:
    *   用于开发或实验与“长度奖励 (length reward)”相关的功能，这通常与强化学习（RL）模型的训练有关。
*   **`remotes/origin/xw/reward-shaping`**:
    *   一个个人开发分支，前缀 `xw/` 很可能是开发者（姓名缩写为 xw）的标识。
    *   分支专注于“奖励塑造 (reward shaping)”功能的开发，这也是强化学习中的一个概念。

### 3. 特殊用途分支

*   **`remotes/origin/adapt_upstream`**:
    *   当一个项目是另一个项目（上游，Upstream）的 fork 时，此分支通常用于同步和适配来自上游仓库的更新，以便保持与原项目的代码同步。
*   **`remotes/origin/det_ver`**:
    *   `det_ver` 可能是 "deterministic version"（确定性版本）的缩写。
    *   此分支可能用于测试或实现一个可复现、行为确定性的版本，这对于调试和评估模型非常重要。
*   **`remotes/origin/gemini-cli-changes`**:
    *   这个分支是为记录和管理通过 Gemini CLI 工具所做的代码修改而创建的。

### 总结

该仓库遵循了标准的 Git 工作流：
- **`main`** 作为稳定的主线。
- **功能分支** (`feature/` 或 `developer/`) 用于并行开发，保持 `main` 分支的整洁。
- **维护性分支** (`adapt_upstream`) 用于项目管理。
