import cv2

def find_camera_ids():
    available_cameras = []
    for i in range(10):  # 0부터 9까지 테스트 (필요하면 범위를 늘릴 수 있음)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    camera_ids = find_camera_ids()
    if camera_ids:
        print("Available camera IDs:", camera_ids)
    else:
        print("No cameras found.")
