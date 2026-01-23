# 如何修改 Git 远程仓库 `origin` 的地址

当你需要将本地的代码仓库推送到一个新的远程仓库时，你需要修改 `origin` 的 URL。

## 步骤

1.  **查看当前的远程仓库地址**
    使用 `git remote -v` 命令可以查看所有配置的远程仓库及其地址。

    ```bash
    git remote -v
    ```
    输出会类似这样：
    ```
    origin  https://github.com/OldUser/OldRepo.git (fetch)
    origin  https://github.com/OldUser/OldRepo.git (push)
    ```

2.  **修改远程仓库地址**
    使用 `git remote set-url` 命令来修改 `origin` 的地址。将 URL 替换成你自己的仓库地址。

    ```bash
    git remote set-url origin https://github.com/YourUser/YourRepo.git
    ```

3.  **验证修改**
    再次运行 `git remote -v` 命令，确认地址已经更新为你自己的仓库地址。

    ```bash
    git remote -v
    ```
    现在输出应该显示新的地址：
    ```
    origin  https://github.com/YourUser/YourRepo.git (fetch)
    origin  https://github.com/YourUser/YourRepo.git (push)
    ```

完成这些步骤后，你就可以使用 `git push` 命令将本地的修改推送到新的远程仓库了。
