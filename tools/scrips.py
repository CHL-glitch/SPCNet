# import pyautogui
# import time
#
# # 启用 FAILSAFE，防止意外情况导致鼠标移动到角落退出
# pyautogui.FAILSAFE = True
#
# # 让用户确认已切换到 CloudCompare
# if pyautogui.confirm('请切换到 CloudCompare 界面，并选中第一个切片。\n准备好后点击 "OK" 开始自动保存。') != 'OK':
#     exit()
#
# # 让用户输入要保存的文件数量，默认10
# try:
#     num_files = int(pyautogui.prompt("请输入要保存的切片数量:", default="10"))
# except ValueError:
#     pyautogui.alert("输入无效，程序终止！")
#     exit()
#
# # 等待用户有足够时间切换窗口
# time.sleep(3)
#
# # 开始批量保存
# for i in range(num_files):
#     try:
#         # 执行保存操作 Ctrl+S
#         pyautogui.hotkey('ctrl', 's')
#         time.sleep(0.8)
#
#         # 输入文件名 slice_x.ply（小写）
#         filename = f"slice_{i}.ply"
#         pyautogui.typewrite(filename)
#         time.sleep(0.5)
#
#         # 确认保存
#         pyautogui.press('enter')
#         time.sleep(0.5)  # 确保文件已保存
#
#         # 按方向键右键一次
#         pyautogui.press('right')
#         time.sleep(0.5)
#
#
#         # 按回车键进入/展开
#         pyautogui.press('enter')
#         time.sleep(0.8)
#
#         # 按方向键下键 (当前文件索引 + 3) 次
#         for _ in range(i + 4):
#             pyautogui.press('down')
#             time.sleep(0.2)  # 适当延迟，避免操作过快
#
#         # 稳定后再继续
#         time.sleep(0.8)
#
#     except Exception as e:
#         pyautogui.alert(f"发生错误: {e}\n程序终止！")
#         exit()
#
# pyautogui.alert("批量保存完成！")
# print("批量保存完成！")
import pyautogui
import time
import os

# 启用 FAILSAFE，防止意外情况导致鼠标移动到角落退出
pyautogui.FAILSAFE = True

# 让用户确认已切换到 CloudCompare
if pyautogui.confirm('请切换到 CloudCompare 界面，并选中第一个切片。\n准备好后点击 "OK" 开始自动保存。') != 'OK':
    exit()

# 让用户输入要保存的文件数量，默认10
try:
    num_files = int(pyautogui.prompt("请输入要保存的切片数量:", default="10"))
except ValueError:
    pyautogui.alert("输入无效，程序终止！")
    exit()

# 指定保存文件夹路径（请根据实际情况修改）
save_folder = "/media/cvmaster/1C0329EB7F42121A/SensatUrban/SensatUrban_Dataset/ply/slice/cambridge_block_2"  # 修改为你的实际保存路径

# 等待用户有足够时间切换窗口
time.sleep(3)


def wait_for_file(file_path, timeout=60):
    """
    等待目标文件存在并且文件大小稳定后返回。
    如果超时则抛出异常。
    """
    start_time = time.time()
    previous_size = -1
    while True:
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            # 当文件大小连续两次检测一致且大于0时，认为保存完成
            if current_size == previous_size and current_size > 0:
                break
            previous_size = current_size
        if time.time() - start_time > timeout:
            raise Exception("文件保存超时")
        time.sleep(1)


# 开始批量保存
for i in range(num_files):
    try:
        # 执行保存操作 Ctrl+S
        pyautogui.hotkey('ctrl', 's')
        time.sleep(0.8)  # 等待保存窗口打开

        # 输入文件名 slice_x.ply（小写）
        filename = f"slice_{i}.ply"
        full_path = os.path.join(save_folder, filename)
        pyautogui.typewrite(filename)
        time.sleep(0.8)

        # 确认保存
        pyautogui.press('enter')

        # 等待文件保存完成

        time.sleep(0.5)  # 稳定后再继续

        # 按方向键右键一次
        pyautogui.press('right')
        time.sleep(0.5)

        # 按回车键进入/展开
        pyautogui.press('enter')
        time.sleep(0.5)
        wait_for_file(full_path)
        # 按方向键下键 (当前文件索引 + 3) 次
        for _ in range(i + 4):
            pyautogui.press('down')
            time.sleep(0.2)

        time.sleep(0.5)

    except Exception as e:
        pyautogui.alert(f"发生错误: {e}\n程序终止！")
        exit()

pyautogui.alert("批量保存完成！")
print("批量保存完成！")
