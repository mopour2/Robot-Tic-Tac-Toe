import cv2, numpy as np, time, random
from dataclasses import dataclass
from typing import List, Optional
from pydobot import Dobot

# ----------------- Dobot setup -----------------
DOBOT_PORT = "/dev/ttyACM0"    # Windows: "COM3" | Linux: "/dev/ttyACM0" or "/dev/ttyUSB0"
TRAVEL_Z=20
HOVER_Z=10
PICK_Z =-45
HOME = (210, 0, TRAVEL_Z, 0)
#-------------------------------------------------
device = Dobot(port=DOBOT_PORT)
device.speed(70, 70)
device.suck(False)
device.move_to(*HOME)
#-------------------------------------------------
PICK_POINTRead = (221, -100)      # pick-up place for RED tokens
PICK_POINTBlue = (259, -100)      # pick-up place for BLUE tokens
humanPickupYNegativ = 0
ComputerPickupYNegativ = 0
PickupYNegativDistanc = 20
CountStable = 100
#-------------------------------------------------
DESTS = {
    "0": (223, -34),
    "1": (223, -0.16),
    "2": (223, 33),
    "3": (260, -34),
    "4": (260, -0.16),
    "5": (260, 33),
    "6": (292, -34),
    "7": (292, -0.16),
    "8": (292, 33),
}
#-------------------------------------------------
DEFAULT_DEST = (329.36, 130.4)

# ----------------- Camera -----------------
cap = cv2.VideoCapture(2)  # camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------- Vision params (HSV thresholds) -----------------
# NOTE: tune these ranges to your lighting
HSV_RED_1 = ((0,   90,  70), (10,  255, 255))
HSV_RED_2 = ((170, 90,  70), (180, 255, 255))
HSV_BLUE  = ((100, 90,  60), (130, 255, 255))
MIN_RATIO = 0.06            # min colored-pixel ratio for a cell to be considered occupied
MORPH_K   = 3               # morphology kernel size for denoising
WARP_SIZE = 300             # warped board image 300×300 → each cell ~100×100

# ================= PASS-BY-REFERENCE STYLE STATE =================
@dataclass
class CamState:
    labels: Optional[List[str]] = None      # latest labels from camera
    prev:   Optional[List[str]] = None      # previous labels for diff
    last_stable: Optional[List[str]] = None # persistent stable state (Last_Stat)
    frame:  Optional[np.ndarray] = None     # raw frame for display
    show:   Optional[np.ndarray] = None     # warped board display image

# ----------------- Window/Display Utilities -----------------
_windows_ready = False

def ensure_windows():
    global _windows_ready
    if not _windows_ready:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Board Read (warped)", cv2.WINDOW_NORMAL)
        _windows_ready = True

def show_frames(state: CamState):
    """Always keep UI responsive; never let windows go black/frozen."""
    ensure_windows()

    if state.frame is None:
        cam_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(cam_img, "No frame...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    else:
        cam_img = state.frame

    if state.show is None:
        warp_img = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)
        cv2.putText(warp_img, "Waiting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    else:
        warp_img = state.show

    cv2.imshow("Camera", cam_img)
    cv2.imshow("Board Read (warped)", warp_img)
    cv2.waitKey(1)  # critical for window event loop

# ----------------- Robot moves -----------------
def pick(x, y):
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0); time.sleep(0.3)
    device.move_to(x, y, PICK_Z,   0); time.sleep(0.2)
    device.suck(True); time.sleep(0.35)
    device.move_to(x, y, HOVER_Z,  0); time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); time.sleep(0.3)


def place(x, y):
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0); time.sleep(0.3)
    device.move_to(x, y, PICK_Z,   0); time.sleep(0.2)
    device.suck(False); time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0); time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); time.sleep(0.3)

# ----------------- Game logic -----------------
def show_board(b):
    print(f"\n {b[0]} | {b[1]} | {b[2]}")
    print("---+---+---")
    print(f" {b[3]} | {b[4]} | {b[5]}")
    print("---+---+---")
    print(f" {b[6]} | {b[7]} | {b[8]}\n")


def winner(b):
    lines = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]
    for a,c,d in lines:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def is_draw(b):
    return all(x != " " for x in b) and winner(b) is None


def computer_move_rule_based(b, comp, human):
    # تابع برای بررسی برنده
    def check_winner(board):
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        for a,c,d in lines:
            if board[a] != " " and board[a] == board[c] == board[d]:
                return board[a]
        return None

    # تابع امتیازدهی برای حالت پایانی
    def evaluate(board):
        w = check_winner(board)
        if w == comp:
            return +1
        elif w == human:
            return -1
        else:
            return 0

    # بررسی خانه‌های خالی
    def empty_cells(board):
        return [i for i, x in enumerate(board) if x == " "]

    # تابع بازگشتی مین‌ماکس
    def minimax(board, depth, is_maximizing):
        w = check_winner(board)
        if w or " " not in board:
            return evaluate(board)

        if is_maximizing:  # نوبت کامپیوتر
            best = -10
            for i in empty_cells(board):
                board[i] = comp
                score = minimax(board, depth + 1, False)
                board[i] = " "
                best = max(best, score)
            return best
        else:  # نوبت انسان
            best = 10
            for i in empty_cells(board):
                board[i] = human
                score = minimax(board, depth + 1, True)
                board[i] = " "
                best = min(best, score)
            return best

    # انتخاب بهترین حرکت برای کامپیوتر
    best_score = -10
    best_move = None
    for i in empty_cells(b):
        b[i] = comp
        score = minimax(b, 0, False)
        b[i] = " "
        if score > best_score:
            best_score = score
            best_move = i

    # در صورتی که هیچ حرکت خاصی نیست، یکی تصادفی
    if best_move is None:
        empties = empty_cells(b)
        best_move = random.choice(empties) if empties else None

    return best_move


# ----------------- Color Board Reader -----------------
_calib_pts = []

def _on_mouse(event, x, y, flags, param):
    global _calib_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(_calib_pts) < 4:
        _calib_pts.append((x, y))

def calibrate_board(cap):
    """Click 4 corners: TL, TR, BR, BL. Enter=OK | c=clear | q=quit"""
    global _calib_pts
    _calib_pts = []
    win = "Board Calib"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    print("[Calib] Click 4 corners TL, TR, BR, BL then press Enter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        for i, p in enumerate(_calib_pts):
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            _calib_pts = []
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None
        elif key in (13, 10):  # Enter
            if len(_calib_pts) == 4:
                src = np.array(_calib_pts, dtype=np.float32)
                dst = np.array([[0, 0], [WARP_SIZE, 0],
                                [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                np.save("calibration_matrix.npy", M)   # ذخیره در فایل
                print("[Calib] Saved calibration to calibration_matrix.npy")
                cv2.destroyWindow(win)
                return M
            else:
                print("[Calib] Please click 4 points!")

def load_calibration():
    """Load saved calibration matrix if available."""
    try:
        M = np.load("calibration_matrix.npy")
        print("[Calib] Loaded existing calibration.")
        return M
    except Exception:
        print("[Calib] No saved file found.")
        return None


def _mask_color(hsv, low, high):
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))


def classify_cell(cell_bgr):
    """cell_bgr: 100x100 BGR → returns 'R', 'B' or ' '"""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    red1 = _mask_color(hsv, *HSV_RED_1)
    red2 = _mask_color(hsv, *HSV_RED_2)
    red  = cv2.bitwise_or(red1, red2)
    blue = _mask_color(hsv, *HSV_BLUE)

    # denoise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    red  = cv2.morphologyEx(red,  cv2.MORPH_OPEN, k)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k)

    #----------------Chenge  حذف نقاط خیلی کوچک ناشی از انعکاس
    red  = cv2.erode(red, k)
    blue = cv2.erode(blue, k)


    # validate saturation/value  ----------------Chenge Light
    s_mask = (hsv[:,:,1] >= 30).astype(np.uint8)*255   #eshba color  70 40
    v_mask = (hsv[:,:,2] >= 30).astype(np.uint8)*255   #shedat nor
    valid  = cv2.bitwise_and(s_mask, v_mask)

    red  = cv2.bitwise_and(red,  valid)
    blue = cv2.bitwise_and(blue, valid)

    r_ratio = red.sum()  / 255.0 / red.size
    b_ratio = blue.sum() / 255.0 / blue.size

    if r_ratio < MIN_RATIO and b_ratio < MIN_RATIO:
        return " "
    return "R" if r_ratio >= b_ratio else "B"


def split_cells(board_bgr):
    """300x300 warped board → list of 9 BGR cells (row-major)."""
    cells = []
    step = WARP_SIZE // 3
    pad  = 6
    for r in range(3):
        for c in range(3):
            y1, y2 = r*step+pad, (r+1)*step-pad
            x1, x2 = c*step+pad, (c+1)*step-pad
            cells.append(board_bgr[y1:y2, x1:x2].copy())
    return cells


def read_board_once(cap, M):
    """
    Returns:
      labels: list of 9 like ['R',' ','B', ...] (index 0..8, top-left to bottom-right)
      frame : raw camera frame
      show  : warped board image with overlays
    """
    ok, frame = cap.read()
    if not ok:
        return None, None, None

    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))
    cells  = split_cells(warped)
    labels = [classify_cell(c) for c in cells]

    show = warped.copy()
    step = WARP_SIZE // 3
    for idx, lab in enumerate(labels):
        r, c = divmod(idx, 3)
        cx, cy = c*step + step//2, r*step + step//2
        color = (0,0,255) if lab=="R" else (255,0,0) if lab=="B" else (200,200,200)
        cv2.putText(show, (lab if lab!=" " else "0"), (cx-10, cy+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.rectangle(show, (c*step, r*step), ((c+1)*step, (r+1)*step), (80,80,80), 1)
    return labels, frame, show


def diff_new_tokens(prev_labels, curr_labels):
    diffs = []
    n = min(len(prev_labels or []), len(curr_labels or []))
    if not prev_labels:
        for i in range(n):
            if curr_labels[i] in ("R","B"):
                diffs.append((i, curr_labels[i]))
        return diffs
    for i in range(n):
        if prev_labels[i] == " " and curr_labels[i] in ("R","B"):
            diffs.append((i, curr_labels[i]))
    if len(curr_labels) > n:
        for i in range(n, len(curr_labels)):
            if curr_labels[i] in ("R","B"):
                diffs.append((i, curr_labels[i]))
    return diffs

# ----------------- Pass-by-ref style human interactions -----------------
def SelectColorHuman(b, cap, M, state: CamState, CountStable=100):
    """Human drops the very first cube to declare color; updates `state` in-place."""
    Flag_Stable = 0
    while True:
        labels_now, frame, show_img = read_board_once(cap, M)
        state.frame, state.show = frame, show_img
        show_frames(state)

        if labels_now is None:
            continue

        diffs = diff_new_tokens(state.prev, labels_now)
       
        Flag_Stable = 0 if diffs else (Flag_Stable + 1)
        
        state.labels = labels_now[:]
        state.prev   = labels_now[:]

        if Flag_Stable >= CountStable:
            for i in range(len(labels_now)):
                if labels_now[i] != " " and b[i] == " ":
                    choice = labels_now[i]
                    #state.last_stable = labels_now[:]  # update Last_Stat equivalent
                    print("------------00-------------------")
                    print(state.last_stable)
                    print(labels_now)
                    print("===============================")
                    diffs2 = diff_new_tokens(state.last_stable, labels_now)
                    DefCount2=len(diffs2) 
                    print(DefCount2)
                    state.last_stable = labels_now[:]  # update Last_Stat equivalent
                    return i, M, choice,DefCount2

#--------------------------------------------------------------------------------------
def human_move_gui(b, cap, M, state: CamState, CountStable=100):
    """Wait for a stable new token; update `state` in-place; return cell index and M."""
    Flag_Stable = 0
    while True:
        labels_now, frame, show_img = read_board_once(cap, M)
        #print(labels_now)
        state.frame, state.show = frame, show_img
        if state.show is not None:
            cv2.putText(state.show, "Your move (place a cube) | c=recalib, q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        show_frames(state)

        if labels_now is None:
            continue

        diffs = diff_new_tokens(state.prev, labels_now)
        Flag_Stable = 0 if diffs else (Flag_Stable + 1)
        state.labels = labels_now[:]
        state.prev   = labels_now[:]

        if Flag_Stable >= CountStable:
           # print("------------2-------------------")
           # print(state.last_stable)
           # print(labels_now)
            #print("===============================")
            if state.last_stable is None:
                state.last_stable = labels_now[:]
                continue
            diffs2 = diff_new_tokens(state.last_stable, labels_now)
            if diffs2:
                i, color = diffs2[0]
                if b[i] == " ":
                    DefCount=len(diffs2) 
                    #print(DefCount)
                    state.last_stable = labels_now[:]
                    return i,color, M,DefCount
                else:
                    cv2.displayOverlay("Board Read (warped)",
                                       f"Cell {i+1} is occupied. Pick another.", 1000)

# ----------------- MAIN -----------------
def main():
    global humanPickupYNegativ, ComputerPickupYNegativ

    # warm-up camera to avoid initial black frames
    for _ in range(10):
        cap.read()
        cv2.waitKey(1)

    b = [" "] * 9
    state = CamState()
    
    first = input("Who starts? (H for Human, C for Computer ,T for Two Human): ").strip().upper()
    
    turn = "human" if first == "H" else "computer"

        
    M = load_calibration()
    if M is None:
        M = calibrate_board(cap)
        if M is None:
            print("Calibration canceled.")
            return


    
    labels0, frame0, show0 = read_board_once(cap, M)
    state.labels = labels0[:] if labels0 else None
    state.prev   = labels0[:] if labels0 else None
    state.last_stable = labels0[:] if labels0 else None
    state.frame  = frame0
    state.show   = show0
    show_frames(state)

    print("------------0-------------------")
    print(state.last_stable)
    print(labels0)
    print("===============================")
   

    if first == "C":
        human, comp = "R", "B"
        PICK_HUMAN = PICK_POINTRead
        PICK_COMP  = PICK_POINTBlue
    else:
        print("Please place the first cube in the desired color.")
        
        i, M, choice,DefCount = SelectColorHuman(b, cap, M, state, CountStable=CountStable)
        if  DefCount>=2 :
           print("****************Wrong*****************")
           return
        print("------------000-------------------")
        print(state.last_stable)
        print(labels0)
        print("===============================")

        b[i] = choice

        if first != "T":
          turn = "computer"
        else :
          turn = "TwoHuman"  

        if choice == "R":
            human, comp = "R", "B"
            PICK_HUMAN = PICK_POINTRead
            PICK_COMP  = PICK_POINTBlue
        else:
            human, comp = "B", "R"
            PICK_HUMAN = PICK_POINTBlue
            PICK_COMP  = PICK_POINTRead

    show_board(b)
    for kk in range(10):
        labels_now, frame, show_img = read_board_once(cap, M)
        #print(labels_now)
        state.frame, state.show = frame, show_img
        if state.show is not None:
            cv2.putText(state.show, "Your move (place a cube) | c=recalib, q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        show_frames(state)


    humanPickupYNegativ = 0
    ComputerPickupYNegativ = 0

    while True:
        if turn == "human":
            print("First  Human plays...")
            i, color1,M ,DefCount= human_move_gui(b, cap, M, state, CountStable=CountStable)
            
            if  DefCount>=2 or color1!=human:
                for kk in range(10):
                    print("****************Wrong*****************")
                break
            #print(DefCount)

            b[i] = human
           
            
        elif turn == "computer":
            print("Computer plays...")
            i = computer_move_rule_based(b, comp,human)
            b[i] = comp
            state.last_stable[i]=comp

            x, y = PICK_COMP
            y = y - ComputerPickupYNegativ
            ComputerPickupYNegativ += PickupYNegativDistanc
            pick(x, y)

            coord = DESTS[str(i)]
            print("[COMP] place ->" )
            place(*coord)
            
            #i, M, choice = get_Next_state(b, cap, M, state, CountStable=CountStable)
        elif turn == "TwoHuman":
            print("Next Human plays...")
            
            i, color1,M ,DefCount= human_move_gui(b, cap, M, state, CountStable=CountStable)
            
            b[i] = comp
            state.last_stable[i]=comp

            if  DefCount>=2 or color1!=comp:
                for kk in range(10):
                    print("****************Wrong*****************")
                break
            #print(DefCount)

            

        show_board(b)
        device.move_to(*HOME)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            M = calibrate_board(cap)
        elif k == ord('q'):
            break

        w = winner(b)
        if w or is_draw(b):
            if w == human:
                print("You win! 🎉")
            elif w == comp and first !="T":
                print("Computer wins! 🤖")
            elif w == comp and first =="T":
                print("Next Human wins! 🤖")    
            else:
                print("Equal 🤝")
            break
        if first != "T":
          turn = "computer" if turn == "human" else "human"
        else :
          turn = "TwoHuman" if turn == "human" else "human"    

if __name__ == "__main__":
    try:
        device.suck(False)
        main()
    finally:
        cap.release(); cv2.destroyAllWindows()
        device.suck(False)
        device.move_to(*HOME)
        device.close()
