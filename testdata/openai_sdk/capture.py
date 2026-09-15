"""mitmdump addon: save completed Chat Completions HTTP body pairs only."""

import json
from pathlib import Path

from mitmproxy import ctx, exceptions


class Capture:
    def load(self, loader):
        loader.add_option("capture_file", str, "", "New JSON body-pair output file")
        self.exchanges = []

    def configure(self, updated):
        if "capture_file" in updated:
            self.path = Path(ctx.options.capture_file)
            try:
                # Reserve a fresh file; never overwrite an approved corpus.
                with self.path.open("x") as output:
                    output.write("[]\n")
            except OSError as error:
                raise exceptions.OptionsError(
                    "capture_file must be a new writable file"
                ) from error

    def response(self, flow):
        if flow.request.method != "POST" or not flow.request.path.endswith(
            "/chat/completions"
        ):
            return
        if flow.response.status_code != 200:
            ctx.log.error("Non-200 response: capture not saved")
            return
        # Mitmproxy buffers by default. content removes HTTP compression without
        # parsing or reserializing JSON/SSE. No headers or SDK objects are saved.
        pair = {
            "request": flow.request.content.decode("utf-8"),
            "response": flow.response.content.decode("utf-8"),
        }
        if (
            "text/event-stream" in flow.response.headers.get("content-type", "")
            and "data: [DONE]" not in pair["response"]
        ):
            ctx.log.error("Incomplete SSE response: capture not saved")
            return
        key = flow.request.headers.get("authorization", "").removeprefix("Bearer ")
        if key and any(key in body for body in pair.values()):
            ctx.log.error("Credential in body: capture not saved")
            return
        self.exchanges.append(pair)
        self.path.write_text(
            json.dumps(self.exchanges, ensure_ascii=False, indent=2) + "\n"
        )


addons = [Capture()]
