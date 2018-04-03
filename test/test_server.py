import os
import unittest

from tornado.testing import AsyncHTTPTestCase

import server

from tornado import testing
from tornado import gen
from tornado import web


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


class TestServerResponse(AsyncHTTPTestCase):

    # must be overridden
    def get_app(self):
        return server.app




if __name__ == '__main__':
    server.test_dir = 'gatherer'
    unittest.main()
