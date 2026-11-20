#!/usr/bin/env python3
"""Turn a Bilibili (or any yt-dlp supported) link into a local file.

Every other script accepts a link directly - this exists so you can inspect the
download step on its own, or pre-fetch a big video while you look at something
else.

    python3 fetch_source.py 'https://www.bilibili.com/video/BV1xx411c7mD' -o wm_work
    python3 fetch_source.py '<link>' --height 720 --cookies ~/cookies.txt

Prints JSON on stdout ({"path", "title", "duration", "height", "cached"}),
progress on stderr. Downloads are cached under
$BILI_WM_MEDIA_DIR (default ~/.cache/bilibili-wm-remover/media), keyed by
url+height, so re-running costs nothing.

Cookies come from --cookies, else $DSH_VIDEO_COOKIES, else the "cookies" field
in ~/.dsh/video.json - the same three sources the DSH video plugin uses, which
is why a machine with DSH already logged in needs no extra setup. Without
cookies, Bilibili still serves <=720p; 1080p+ needs a signed-in session.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="a URL, or a local file to pass through")
    ap.add_argument("-o", "--out", default=None,
                    help="where to record the resolved path (writes resolved.json); "
                         "the media itself stays in the cache")
    C.add_fetch_args(ap)
    a = ap.parse_args()

    if not C.is_url(a.src):
        if not os.path.exists(a.src):
            sys.exit("input not found: %s" % a.src)
        info = {"path": os.path.abspath(a.src), "title": None, "duration": None,
                "height": a.height, "url": None, "cached": True}
    else:
        info = C.fetch_video(a.src, height=a.height, cookies=a.cookies,
                             force=a.force_download)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "resolved.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
