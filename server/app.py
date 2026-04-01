from __future__ import annotations

import uvicorn

from server.main import HOST, PORT


def main() -> None:
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()