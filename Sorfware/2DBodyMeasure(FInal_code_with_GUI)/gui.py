from pathlib import Path
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, ttk
import cv2
from PIL import Image, ImageTk, ImageDraw
import os
import u2net
import torch
from measure_model import Conv_BoDiEs, predict_and_save
import csv
from screeninfo import get_monitors
import tkinter as tk
import shutil
import os


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")

# 성별, 모델 초기화
gender = 0
model_path = ""

# global로 close_camera를 선언
close_camera_func = None
cap = 0

# GUI 변수 전역 설정정
canvas = entry_1 = button_1 = button_2 = button_3 = None
text_name = text_gender = text_welcome = text_measurement = None
text_male = text_female = text_camera = text_stand_pose = None
image_2 = image_3 = image_4 = image_5 = None
text_countdown = text_captured = entry_bg_1 = tree = result_foler= image_folder = None


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def male_button_clicked():
    """
    Male 버튼 클릭 시 실행: 성별을 남성으로 설정, 버튼 이미지 변경
    동시에 'female' 버튼의 이미지를 초기화
    """
    global gender, model_path
    
    # 남성용 모델 경로 설정 및 성별 값 할당
    model_path = "bodymodel/male/Conv_BoDiEs_male_grayscale.pth"
    gender = 1
    print(f"Gender: Male {gender}")
    
    # Male 버튼에 체크 이미지 적용
    new_button_image = PhotoImage(file=relative_to_assets("button_check.png"))
    button_2.config(image=new_button_image)
    button_2.image = new_button_image  # 이미지 참조 유지
    
    # Female 버튼 이미지를 기본 이미지로 리셋
    new_button_image = PhotoImage(file=relative_to_assets("button_empty.png"))
    button_3.config(image=new_button_image)
    button_3.image = new_button_image  # 이미지 참조 유지

def female_button_clicked():
    """
    'Female' 버튼 클릭 처리: 성별을 여성으로 설정/ 해당 버튼의 이미지를 업데이트
    동시에 'Male' 버튼의 이미지를 초기화
    """
    global gender, model_path  # 성별과 모델 경로를 위한 전역 변수
    
    model_path = "bodymodel/female/Conv_BoDiEs_female_grayscale.pth"  # 여성 모델 경로
    gender = 2  # 성별 설정
    print(f"Gender: Female {gender}")
    
    # 'Female' 버튼에 체크 이미지 적용
    new_button_image = PhotoImage(file=relative_to_assets("button_check.png"))
    button_3.config(image=new_button_image)
    button_3.image = new_button_image  # 이미지 참조 유지
    
    # 'Male' 버튼에 기본 이미지 적용
    new_button_image = PhotoImage(file=relative_to_assets("button_empty.png"))
    button_2.config(image=new_button_image)
    button_2.image = new_button_image  # 이미지 참조 유지


def show_camera(canvas, image_2):
    global cap
    cap = cv2.VideoCapture(0)  # 기본 카메라 시작

    if not cap.isOpened():  # 카메라가 열리지 않는 경우
        print("Camera not found!")
        return

    def update_frame():
        ret, frame = cap.read()  # 한 프레임 읽기
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 화면 회전
            
            # image_2의 크기에 맞게 리사이즈
            bbox = canvas.bbox(image_2)
            image_width = bbox[2] - bbox[0]
            image_height = bbox[3] - bbox[1]
            frame_resized = cv2.resize(frame, (image_width, image_height))

            # 둥근 모서리를 추가한 이미지 생성
            image = Image.fromarray(frame_resized)
            image_with_rounded_corners = add_rounded_corners(image, radius=40)
            photo = ImageTk.PhotoImage(image=image_with_rounded_corners)

            # 캔버스 이미지 업데이트
            canvas.itemconfig(image_2, image=photo)
            canvas.image = photo  # 참조 유지

        canvas.after(10, update_frame)  # 10ms마다 프레임 갱신
    
    update_frame()  # 첫 프레임 업데이트 시작

    def close_camera():
        cap.release()  # 카메라 해제

    return close_camera  # 종료 함수 반환


def add_rounded_corners(image, radius):
    # 이미지의 크기
    width, height = image.size
    
    # 마스크 이미지 만들기 (흰색 배경)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 둥근 모서리를 위한 원을 그리기
    draw.rounded_rectangle((0, 0, width, height), radius, fill=255)
    
    # 마스크를 이미지에 적용
    image.putalpha(mask)
    
    return image

def capture_frame():
    # 카메라 프레임을 캡처하고 저장하는 부분
    global cap
    ret, frame = cap.read()
    if ret:
        filename = "result/image/captured_image.png"
        cv2.imwrite(filename, frame)  # 이미지를 파일로 저장
    cap.release()

def on_start_button():
    """
    시작 버튼 클릭 시 실행:
    - 사용자 입력으로 폴더 생성
    - UI 요소 숨김
    - 카메라 실행 및 카운트다운 시작
    """
    global gender, close_camera_func, canvas, entry_1, button_1, button_2, button_3
    global text_name, text_gender, text_welcome, text_measurement, text_male
    global text_female, image_2, text_camera, image_4, text_stand_pose
    global text_countdown, text_captured, image_5, entry_bg_1, result_folder, image_folder

    # Entry에서 사용자 입력 가져오기
    value = entry_1.get()
    base_path = value if value.strip() != "" else "no_id"  # 입력 없으면 기본값 설정
    print(f"Hello {value if value.strip() != '' else 'no_id'}")

    # 폴더 경로 설정
    result_folder = os.path.join(base_path, 'result')
    image_folder = os.path.join(base_path, 'image')

    # 폴더 생성 함수
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")

    # 폴더 구조 생성
    create_folder(base_path)
    create_folder(result_folder)
    create_folder(image_folder)
    print(f"Created folder structure: {base_path}/result and {base_path}/image")

    # 성별이 설정된 경우에만 실행
    if gender == 1 or gender == 2:
        print("Start")

        # 불필요한 UI 요소 숨기기
        entry_1.place_forget()
        button_1.place_forget()
        button_2.place_forget()
        button_3.place_forget()
        canvas.itemconfig(text_name, state="hidden")
        canvas.itemconfig(text_gender, state="hidden")
        canvas.itemconfig(text_welcome, state="hidden")
        canvas.itemconfig(text_measurement, state="hidden")
        canvas.itemconfig(text_male, state="hidden")
        canvas.itemconfig(text_female, state="hidden")
        canvas.itemconfig(entry_bg_1, state="hidden")

        # 카메라 시작
        canvas.itemconfig(image_2, state="normal")
        canvas.itemconfig(text_camera, state="normal")
        close_camera_func = show_camera(canvas, image_2)

        # "Stand in front pose" 텍스트 표시 (3초 후 카운트다운 시작)
        canvas.itemconfig(image_4, state="normal")
        canvas.itemconfig(text_stand_pose, state="normal")
        canvas.after(3000, start_countdown)

def start_countdown():
    """
    카운트다운 시작 함수:
    - 5초 카운트다운을 표시
    - 종료 후 "Captured" 텍스트를 보여주고 프레임을 캡처 및 처리
    """
    global text_stand_pose, text_countdown, text_captured, canvas

    # "Stand in front pose" 텍스트 숨기기
    canvas.itemconfig(text_stand_pose, state="hidden")
    countdown_time = 5  # 카운트다운 시작 시간 (5초)

    def countdown():
        nonlocal countdown_time
        if countdown_time > 0:
            # 현재 카운트다운 값을 화면에 표시
            canvas.itemconfig(text_countdown, text=str(countdown_time))
            canvas.itemconfig(text_countdown, state="normal")  # 텍스트 보이기
            countdown_time -= 1  # 시간 감소
            canvas.after(1000, countdown)  # 1초 후 재귀 호출
        elif countdown_time == 0:
            # 카운트다운 종료 시 텍스트 숨기기
            canvas.itemconfig(text_countdown, state="hidden")
            
            # "Captured" 텍스트 표시
            canvas.itemconfig(text_captured, state="normal")

            countdown_time -= 1  # 추가 상태 처리
            canvas.after(100, countdown)  # 짧은 지연 후 재귀 호출
        else:
            # 카메라에서 프레임 캡처 및 저장
            capture_frame()
            # 카메라 종료
            close_camera_func()
            # 이미지 처리 함수 호출
            image_processing()

    countdown()  # 카운트다운 시작

def image_processing():
    """
    이미지 처리 함수:
    - 이미지를 읽어 90도 반시계 방향으로 회전 후 저장
    - U2Net 모델을 사용해 이미지 전처리 (세그멘테이션 수행)
    - 처리 완료 후 `estimation` 함수 호출
    """
    # 저장된 이미지 파일 읽기
    filename = 'result/image/captured_image.png'
    need2Rotateimage = cv2.imread(filename)

    # 이미지 90도 반시계 방향 회전
    rotated_image_clockwise = cv2.rotate(need2Rotateimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(filename, rotated_image_clockwise)  # 회전된 이미지 저장

    # 이미지 경로 설정
    image_dir = os.path.join(os.getcwd(), 'result', 'image')  # 입력 이미지 경로
    prediction_dir = os.path.join(os.getcwd(), 'result', 'grayscale_image')  # 출력 경로
    model_path2 = os.path.join(os.getcwd(), 'u2net_saved_models', 'u2net_human_seg', 'u2net_human_seg.pth')  # U2Net 모델 경로

    # U2Net 모델을 사용한 이미지 세그멘테이션 실행
    u2net.run_u2net_inference(image_dir, prediction_dir, model_path2, num_images=2)

    print("Image Processing Complete\n\n")

    # 후속 처리 실행
    estimation()

def estimation():
    """
    이미지 기반 예측 함수:
    - CUDA 또는 CPU를 확인해 모델을 초기화
    - 마스크 이미지와 원본 이미지 결합 (AND 연산)
    - 모델을 사용해 예측을 수행하고 결과를 저장
    - 결과를 화면에 표시
    """
    global model_path

    # 장치 설정 (CUDA 사용 여부 확인)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    if torch.cuda.is_available():
        print(f'Device name: {torch.cuda.get_device_name(0)}')

    # 모델 초기화 및 장치로 이동
    model = Conv_BoDiEs().to(device)

    # 이미지 및 마스크 이미지 불러오기
    image = cv2.imread('result/image/captured_image.png', cv2.IMREAD_GRAYSCALE)  # 원본 이미지
    mask_image = cv2.imread('result/grayscale_image/captured_image.png', cv2.IMREAD_GRAYSCALE)  # 마스크 이미지

    # 원본 이미지와 마스크 이미지 AND 연산 -> 배경 제거된 이미지 생성
    black = cv2.bitwise_and(image, mask_image)
    cv2.imwrite("result/black_background.png", black)  # 결과 이미지 저장

    # 모델을 사용해 이미지 예측 및 결과 저장
    predict_and_save(
        model, 
        model_path, 
        image_path="result/grayscale_image/captured_image.png",
        result_file_path='result/predicted/results.csv',
        device=device
    )

    # 결과 화면에 표시
    show_result()

def show_result():
    """
    결과 화면 표시 함수:
    - 이미지와 텍스트를 숨기고 결과 이미지를 표시
    - CSV 파일의 데이터를 읽어 Treeview에 표시
    """
    global canvas, image_2, text_camera, text_captured, image_4, image_5, image_3, tree
    global result_folder, image_folder

    # 불필요한 요소 숨기기
    canvas.itemconfig(image_2, state="hidden")
    canvas.itemconfig(text_camera, state="hidden")
    canvas.itemconfig(text_captured, state="hidden")
    canvas.itemconfig(image_4, state="hidden")

    # 결과 이미지 표시
    canvas.itemconfig(image_5, state="normal")
    canvas.itemconfig(image_3, state="normal")

    # 결과 이미지 크기 조정 및 둥근 모서리 적용
    bbox = canvas.bbox(image_3)
    image_width, image_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    image_or = cv2.imread("result/grayscale_image/captured_image.png")
    resized_image = cv2.resize(image_or, (image_width, image_height))
    imageRe = Image.fromarray(resized_image)
    new_image_tk = ImageTk.PhotoImage(add_rounded_corners(imageRe, radius=40))
    canvas.itemconfig(image_3, image=new_image_tk)
    canvas.image = new_image_tk

    # CSV 데이터 읽기
    csv_file_path = "result/predicted/results.csv"
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile))

    # Treeview 설정
    tree_frame = tk.Frame(canvas)
    tree_frame.place(x=100, y=770)
    tree = ttk.Treeview(tree_frame, columns=[str(i) for i in range(len(data[0]))], show="headings")

    # 스타일 설정
    style = ttk.Style()
    style.configure("Treeview", font=("Assistant", 14, "bold"), rowheight=30)
    style.configure("Treeview.Heading", font=("Assistant", 16, "bold"), anchor="center")

    tree.config(height=4)

    # 헤더 설정
    for col_num, col_name in enumerate(data[0]):
        tree.heading(col_num, text=col_name)

    # 데이터 삽입 및 숫자 형식 정리
    for row in data[1:]:
        row[1] = str(int(float(row[1]))) if row[1].replace('.', '', 1).isdigit() else row[1]
        tree.insert("", "end", values=row)

    # 행 색상 설정 (짝수, 홀수)
    for index, row in enumerate(tree.get_children()):
        tree.item(row, tags="even" if index % 2 == 0 else "odd")
    tree.tag_configure("even", background="#f0f0f0")
    tree.tag_configure("odd", background="#ffffff")

    # Treeview 표시
    tree.pack()

    # save to user folder

    source1 = "result/grayscale_image/captured_image.png"
    destination1 = image_folder
    shutil.copy(source1, destination1)

    source2 = "result/predicted/results.csv"
    destination2 = result_folder
    shutil.copy(source2, destination2)
    
    print("파일이 성공적으로 복사되었습니다.")



def create_window():
    global canvas, entry_1, button_1, button_2, button_3, text_name, text_gender, text_welcome, text_measurement, text_male, text_female, image_2, text_camera, image_4, text_stand_pose, text_countdown, text_captured, image_5, entry_bg_1, image_3, button_4


    # Tkinter 기본 창 생성
    window = Tk()

    # 듀얼 스크린 정보 가져오기
    monitors = get_monitors()

    # 확장된 디스플레이(두 번째 모니터)를 선택
    if len(monitors) > 1:
        second_monitor = monitors[1]  # 두 번째 화면 정보 가져오기
    else:
        second_monitor = monitors[0]  # 두 번째 화면이 없으면 첫 번째 화면 사용

    # 두 번째 모니터 위치
    screen_x = second_monitor.x
    screen_y = second_monitor.y
    screen_width = second_monitor.width
    screen_height = second_monitor.height


    # 창 크기 설정 및 위치 지정
    window.geometry(f"{screen_width}x{screen_height}+{screen_x}+{screen_y}")
    window.configure(bg="#FFFFFF")

    # 창을 화면 경계로 이동 후 전체화면 설정
    def enter_fullscreen():
        window.attributes("-fullscreen", True)

    # ESC 키를 눌렀을 때 전체화면 이동됨
    def exit_fullscreen(event=None):
        window.attributes("-fullscreen", False)

    # ESC 키와 연결
    window.bind("<Escape>", exit_fullscreen)

    # 초기 화면 설정
    window.after(100, enter_fullscreen)  # 0.1초 지연 후 전체화면 진입

    canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 1024,
        width = 600,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)

    image_image_1 = PhotoImage(
        file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(
        300.00006103515625,
        511.00000858148496,
        image=image_image_1
    )

    #----------- 시작 페이지 텍스트 welcome ~~ -------#
    text_welcome = canvas.create_text(
        60.0,
        64.0,
        anchor="nw",
        text="Welcome !",
        fill="#2F275E",
        font=("Assistant ExtraBold", 96 * -1)
    )

    text_measurement = canvas.create_text(
        70.0,
        190.0,
        anchor="nw",
        text="2D Base Body\nMeasurement",
        fill="#554E80",
        font=("Assistant SemiBold", 48 * -1)
    )
    #-------------------------------------------------------------#

    # Name 텍스트트
    text_name = canvas.create_text(
        88.0,
        390.0,
        anchor="nw",
        text="Name",
        fill="#554E80",
        font=("Assistant Bold", 36 * -1)
    )

    # Gender 텍스트 
    text_gender = canvas.create_text(
        88.0,
        620.0,
        anchor="nw",
        text="Gender",
        fill="#554E80",
        font=("Assistant Bold", 36 * -1)
    )

    # 카운트다운 및 capture 등 텍스트 표시되는 부분분
    image_image_4 = PhotoImage(file=relative_to_assets("entry_2_round.png"))
    image_4 = canvas.create_image(
        300.0,  
        920.0,  
        image= image_image_4,
        state="hidden"  
    )

    # 'Stand in front pose' 텍스트  
    text_stand_pose = canvas.create_text(
        300.0,  
        920.0,  
        anchor="center",  
        text="Stand in front pose",  
        fill="#2F275E",  
        font=("Assistant SemiBold", 36 * -1),
        state="hidden" 
    )

    # 카운트다운 텍스트 정의 
    text_countdown = canvas.create_text(
        300.0,  
        920.0,  
        anchor="center",  
        text="5",  # 초기 카운트다운 숫자
        fill="#2F275E",  
        font=("Assistant SemiBold", 48 * -1),
        state="hidden"  
    )

    # 'Captured' 텍스트 정의 
    text_captured = canvas.create_text(
        300.0,  
        920.0,  
        anchor="center", 
        text="Captured",  
        fill="#2F275E",  
        font=("Assistant ExtraBold", 48 * -1),
        state="hidden" 
    )

    # Name 칸에 입력
    entry_image_1 = PhotoImage(
        file=relative_to_assets("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        290.0,
        485.0,
        image=entry_image_1
    )
    entry_1 = Entry(
        bd=0,
        bg="#F2F2F2",
        fg="#000716",
        highlightthickness=0
    )
    entry_1.place(
        x=88.0,
        y=450.0,
        width=404.0,
        height=68.0
    )

    # 시작 버튼튼
    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command= on_start_button,
        relief="flat"
    )
    button_1.place(
        x=218.0,
        y=881.0,
        width=165.0,
        height=50.0
    )

    # 카메라 화면 보여지는 부분분
    image_image_2 = PhotoImage(
        file=relative_to_assets("image_2.png"))
    image_2 = canvas.create_image(
        300.0,
        450.0,
        image=image_image_2,
        state = "hidden"
    )

    image_image_5 = PhotoImage(
        file=relative_to_assets("entry_3_R2.png"))
    image_5 = canvas.create_image(
        300.0,
        850.0,  # 890 originally
        image=image_image_5,
        state = "hidden"
    )

    # 카메라 켜짐 텍스트 
    text_camera = canvas.create_text(
        200.0,
        0.0,
        anchor="nw",
        text="Camera On",
        fill="#AF1740",
        font=("Assistant ExtraBold", 40 * -1),
        state = "hidden"
    )

    # 최종 결과 화면 
    image_image_3 = PhotoImage(
        file=relative_to_assets("image_3.png"))
    image_3 = canvas.create_image(
        300.0,
        380.0,  # originally 405
        image=image_image_3,
        state = "hidden"
    )

    # male 기본 버튼 
    button_image_2 = PhotoImage(
        file=relative_to_assets("button_empty.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command= male_button_clicked,
        relief="flat"
    )
    button_2.place(
        x=88.0,
        y=680.0,  
        width=30.0,
        height=30.0
    )

    # female 기본 버튼
    button_image_3 = PhotoImage(
        file=relative_to_assets("button_empty.png"))
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command= female_button_clicked,
        relief="flat",
    )
    button_3.place(
        x=285.0,
        y=680.0,
        width=30.0,
        height=30.0
    )

    # male 텍스트 
    text_male =canvas.create_text(
        135.0,
        680.0,
        anchor="nw",
        text="Male",
        fill="#557ECC",
        font=("Assistant Bold", 24 * -1),
        
    )

    # female 텍스트
    text_female = canvas.create_text(
        332.0,
        680.0,
        anchor="nw",
        text="Female",
        fill="#557ECC",
        font=("Assistant Bold", 24 * -1),
    )

    # 창조정 가능
    window.resizable(True, True)
    window.mainloop()


if __name__ == "__main__":
    create_window()