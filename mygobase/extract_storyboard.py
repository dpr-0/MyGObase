import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta

# from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import ass  # type: ignore
import cv2

# import ffmpeg
# from ffmpeg import Stream
# from PIL import Image
from tqdm import tqdm  # type: ignore

# from mygobase import model

FRAME_TIME = 24000 / 1001
type FrameNumber = int
type Subtitle = str


@dataclass
class Storyboard:
    frame_number: FrameNumber
    subtitle: Subtitle
    picture: bytes


@dataclass
class Episode:
    id: int
    storyboards: List[Storyboard]


def find_ep_id(filename: str) -> int:
    name = Path(filename).stem
    name = name.replace("[Nekomoe kissaten] BanG Dream! It’s MyGO!!!!! ", "").replace(
        "[BDRip].JPTC", ""
    )
    start_idx = name.find("[") + 1
    end_idx = name.find("]")
    id = int(name[start_idx:end_idx])
    return id


# def extract_frame(stream: Stream, frame_num: int):
#     while isinstance(stream, ffmpeg.nodes.OutputStream):
#         stream = stream.node.incoming_edges[0].upstream_node.stream()
#     out, _ = (
#         stream.filter_("select", "gte(n,{})".format(frame_num))  # type: ignore
#         .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1)
#         .run(capture_stdout=True, capture_stderr=True)
#     )

#     # return Image.fromarray(np.frombuffer(out, np.uint8).reshape([height, width, 3]))
#     # image = Image.frombytes("RGB", (1920, 1080), out)
#     # image.save(f"{frame_num}.png")
#     return out


# def extrac_picture(filename: str, frame_nums: List[FrameNumber]) -> List[bytes]:
#     stream = ffmpeg.input(filename)
#     res = []
#     for fn in tqdm(frame_nums):
#         res.append(extract_frame(stream, fn))
#     return res


def extrac_picture(filename: str, frame_nums: List[FrameNumber]) -> List[bytes]:
    fn_table = {frame_num: frame_num for frame_num in frame_nums}
    frame_num = -1
    res = []
    print(f"start read:{filename}")
    cap = cv2.VideoCapture(filename)
    print("read finish")
    with tqdm(total=len(fn_table)) as pbar:
        while cap.isOpened():
            frame_num += 1
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num not in fn_table:
                continue
            frame_bytes = cv2.imencode(".png", frame)[1].tobytes()
            # image = Image.open(BytesIO(frame_bytes))
            # image.save(f"{frame_num}.png")
            res.append(frame_bytes)
            pbar.update(1)
            # break
    return res


def clear_subtitle(subtitle: str) -> str:
    if "{" in subtitle:  # type: ignore
        end = subtitle.find("}")
        assert end > -1
        subtitle = subtitle[end + 1 :]
    return subtitle


def extract_fn_and_sub(filename: str) -> List[Tuple[FrameNumber, Subtitle]]:
    with open(filename, encoding="utf_8_sig") as f:
        doc = ass.parse(f)
    storyboards: List[Tuple[FrameNumber, Subtitle]] = []
    for d in doc.events:
        d: ass.Dialogue  # type: ignore
        if d.style != "Dial_CH":
            continue

        # find frame_number
        start_time: timedelta = d.start  # type: ignore
        end_time: timedelta = d.end  # type: ignore
        key_time = start_time + (end_time - start_time) / 2
        frame_number = round(key_time.total_seconds() * FRAME_TIME)

        # find sub
        subtitle = clear_subtitle(str(d.text))
        storyboards.append((frame_number, subtitle))
    return storyboards


def extract_storyboard(epid: int, video_filename: str, sub_filename: str) -> Episode:
    print((video_filename, sub_filename))
    fn_sub_pairs = extract_fn_and_sub(sub_filename)
    frame_numbers = list(map(lambda x: x[0], fn_sub_pairs))
    pcis = extrac_picture(video_filename, frame_numbers)
    storyboards: List[Storyboard] = []
    for (fn, sub), pic in zip(fn_sub_pairs, pcis):
        storyboards.append(Storyboard(frame_number=fn, subtitle=sub, picture=pic))
    episode = Episode(id=epid, storyboards=storyboards)
    return episode


def store_db(ep: Episode):
    epid = ep.id
    data = [(epid, sb.frame_number, sb.subtitle, sb.picture) for sb in ep.storyboards]
    with sqlite3.connect("db/mygo.db") as conn:
        try:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO episodes (episode, frame_number, subtitle, picture) VALUES (?,?,?,?)",
                data,
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            print(e)


if __name__ == "__main__":
    video_filenames = []
    for dirpath, _, filenames in Path("./videos").walk():
        for filename in filenames:
            filename = (dirpath / filename).as_posix()
            video_filenames.append(filename)
    video_filenames.sort(key=lambda filename: int(Path(filename).stem))

    sub_filenames = []

    for dirpath, _, filenames in Path("./sub").walk():
        for i, filename in enumerate(filenames):
            filename = (dirpath / filename).as_posix()
            sub_filenames.append(filename)
    sub_filenames.sort(key=find_ep_id)

    episodes: List[Episode] = []
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for epid, (video_filename, sub_filename) in enumerate(
            zip(video_filenames, sub_filenames), start=1
        ):
            futures.append(
                executor.submit(extract_storyboard, epid, video_filename, sub_filename)
            )
    for future in as_completed(futures):
        assert not future.cancelled()
        episodes.append(future.result())
    print(f"Extracted {len(episodes)} episodes")
    for episode in episodes:
        store_db(episode)
