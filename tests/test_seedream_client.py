import json, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from adapters.image.seedream_client import SeedreamClient, SeedreamConfig, SeedreamError

class Fake:
    def __init__(self, responses, payload=b"PNGDATA"):
        self.responses = list(responses); self.calls = []; self.payload = payload
    def transport(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body.decode())))
        return self.responses.pop(0)
    def download(self, url, timeout):
        return self.payload

def client(fake):
    return SeedreamClient(SeedreamConfig(api_key="k"), transport=fake.transport, download=fake.download)

class SeedreamClientTest(unittest.TestCase):
    def test_generate_writes_file_and_sends_body(self):
        with TemporaryDirectory() as d:
            f = Fake([(200, json.dumps({"data":[{"url":"https://x/img.png"}]}).encode())])
            out = client(f).generate("a cat", Path(d)/"o.png")
            self.assertEqual(out.read_bytes(), b"PNGDATA")
            url, headers, body = f.calls[0]
            self.assertTrue(url.endswith("/images/generations"))
            self.assertEqual(headers["Authorization"], "Bearer k")
            self.assertEqual(body["model"], "doubao-seedream-5.0-pro")
            self.assertFalse(body["watermark"])
            self.assertEqual(body["output_format"], "png")

    def test_retries_transient(self):
        with TemporaryDirectory() as d:
            f = Fake([(500, b"oops"), (200, json.dumps({"data":[{"url":"u"}]}).encode())])
            client(f).generate("x", Path(d)/"o.png")
            self.assertEqual(len(f.calls), 2)

    def test_size_fallback(self):
        with TemporaryDirectory() as d:
            f = Fake([(400, b"bad size"), (200, json.dumps({"data":[{"url":"u"}]}).encode())])
            client(f).generate("x", Path(d)/"o.png", size="2048x1365")
            self.assertEqual(f.calls[0][2]["size"], "2048x1365")
            self.assertEqual(f.calls[1][2]["size"], "1024x1024")

    def test_error_raises(self):
        with TemporaryDirectory() as d:
            f = Fake([(400, b"bad prompt")])
            with self.assertRaises(SeedreamError):
                client(f).generate("x", Path(d)/"o.png")

if __name__ == "__main__":
    unittest.main()
