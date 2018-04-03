import os
import json
import unittest
import urllib

from tornado.testing import AsyncHTTPTestCase

import server

from tornado import testing
from tornado import gen
from tornado import web


# https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-a-python-script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# python3 -m unittest -v test.testserver

class TestServerBoot(AsyncHTTPTestCase):

    # Must be overridden 
    def get_app(self):
        return server.app

    def test_get_index(self):
        res = self.fetch('/')
        self.assertEqual(res.code, 200)

    def test_get_static_js(self):
        res = self.fetch('/client/js/index.js')
        self.assertEqual(res.code, 200)

    def test_get_static_css(self):
        res = self.fetch('/client/style.css')
        self.assertEqual(res.code, 200)

    def test_get_static_html(self):
        res = self.fetch('/client/index.html')
        self.assertEqual(res.code, 200)






# http://www.tornadoweb.org/en/stable/testing.html
# Important to understand that with an async library/server it's necessary to test in an async way
# TLDR, if you experience undefined behaviour, double check the use of yield, self.stop, self.wait()

class TestServerResponse(AsyncHTTPTestCase):

    # must be overridden
    def get_app(self):
        return server.app

    def test_post_api(self):
        data = json.load(open(os.path.join(__location__, 'kp.json')))
        resp = self.fetch(
            '/api',
            method='POST',
            body=urllib.parse.urlencode(data),
            follow_redirects=False
        )

        #print("\n\n\n", resp, "\n\n\n")
        self.assertEqual(resp.code, 200)
    
    def test_get_api_code(self):
        res = self.fetch('/api')
        self.assertEqual(res.code, 200)
    
    def test_get_api_content(self):
        res = self.fetch('/api')
        self.assertIn(res.body, 'uuid')
        self.assertIn(res.body, 'equation')
    

class TestWebSocketBoot(testing.AsyncTestCase):
    pass



if __name__ == '__main__':
    server.test_dir = 'gatherer'
    unittest.main()
