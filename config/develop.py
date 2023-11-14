#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/04/13, ZJUICSR'

import os

''' GENERATE SECRET KEY '''
if not os.environ.get('SECRET_KEY'):
    # Attempt to read the secret from the secret file
    # This will fail if the secret has not been written
    try:
        with open('.secai_key', 'rb') as secret:
            key = secret.read()
    except (OSError, IOError):
        key = None

    if not key:
        key = os.urandom(64)
        # Attempt to write the secret file
        # This will fail if the filesystem is read-only
        try:
            with open('.secai_key', 'wb') as secret:
                secret.write(key)
                secret.flush()
        except (OSError, IOError):
            print("Write file error for read-only system!")
            exit(1)


''' SERVER SETTINGS '''
class Config(object):
    ROOT_FOLDER = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    STATIC_FOLDER = "{:s}/../web/static".format(os.path.dirname(os.path.abspath(__file__)))

    # redis config, evnironment `SECAI_STABLE` is the stable version
    REDIS_PASS = "zjuicsr123"
    REDIS_HOST = "10.15.201.50"
    REDIS_PORT = 26379
    REDIS_DB = 2
    REDIS_URL = "redis://{:s}:{:d}".format(REDIS_HOST, REDIS_PORT)

    '''
    SECRET_KEY is the secret value used to creation sessions and sign strings. This should be set to a random string. In the
    interest of ease, SecAI will automatically create a secret key file for you. If you wish to add this secret key to
    your instance you should hard code this value to a random static value.

    You can also remove .secai_key from the .gitignore file and commit this file into whatever repository
    you are using: http://flask.pocoo.org/docs/0.11/quickstart/#sessions
    '''
    SECRET_KEY = os.environ.get('SECRET_KEY') or key

    '''
    SQLALCHEMY_TRACK_MODIFICATIONS is automatically disabled to suppress warnings and save memory. You should only enable
    this if you need it.
    '''
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    '''
    SESSION_TYPE is a configuration value used for Flask-Session. It is currently unused in SecAI.
    http://pythonhosted.org/Flask-Session/#configuration
    '''
    SESSION_TYPE = "filesystem"

    '''
    SESSION_FILE_DIR is a configuration value used for Flask-Session. It is currently unused in SecAI.
    http://pythonhosted.org/Flask-Session/#configuration
    '''
    SESSION_FILE_DIR = "/tmp/flask_session"

    '''
    SESSION_COOKIE_HTTPONLY controls if cookies should be set with the HttpOnly flag.
    '''
    SESSION_COOKIE_HTTPONLY = True

    '''
    PERMANENT_SESSION_LIFETIME is the lifetime of a session.
    '''
    PERMANENT_SESSION_LIFETIME = 604800  # 7 days in seconds

    '''
    HOST specifies the hostname where the SecAI instance will exist. It is currently unused.
    '''
    HOST = "icsr.zju.edu.cn"

    '''
    MAILFROM_ADDR is the email address that emails are sent from if not overridden in the configuration panel.
    '''
    MAILFROM_ADDR = "noreply@zju.edu.cn"

    '''
    LOG_FOLDER is the location where logs are written
    These are the logs for SecAI key submissions, registrations, and logins
    The default location is the SecAI/logs folder
    '''
    LOG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs') or os.environ.get('LOG_FOLDER')

    '''
   OUTPUT_FOLDER is the location where middle files are written
   '''
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output') or os.environ.get('LOG_FOLDER')

    '''
    UPLOAD_FOLDER is the location where files are uploaded.
    The default destination is the SecAI/uploads folder.
    '''
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    'uploads')

    '''
    TEMPLATES_AUTO_RELOAD specifies whether Flask should check for modifications to templates and
    reload them automatically
    '''
    TEMPLATES_AUTO_RELOAD = True

    '''
    TRUSTED_PROXIES defines a set of regular expressions used for finding a user's IP address if the SecAI instance
    is behind a proxy. 

    SecAI only uses IP addresses for cursory tracking purposes. It is ill-advised to do anything complicated based
    solely on IP addresses.
    '''
    TRUSTED_PROXIES = [
        '^127\.0\.0\.1$',
        # Remove the following proxies if you do not trust the local network
        '^::1$',
        '^fc00:',
        '^10\.',
        '^172\.(1[6-9]|2[0-9]|3[0-1])\.',
        '^192\.168\.'
    ]

    '''
    CACHE_TYPE specifies how SecAI should cache configuration values. If CACHE_TYPE is set to 'redis', SecAI will make use
    of the REDIS_URL specified in environment variables. You can also choose to hardcode the REDIS_URL here.

    It is important that you specify some sort of cache as SecAI uses it to store values received from the database.

    CACHE_REDIS_URL is the URL to connect to Redis server.
    Example: redis://user:password@localhost:6379

    http://pythonhosted.org/Flask-Caching/#configuring-flask-caching
    '''


    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    if CACHE_REDIS_URL:
        CACHE_TYPE = 'redis'
    else:
        CACHE_TYPE = 'simple'

    '''
    UPDATE_CHECK specifies whether or not SecAI will check whether or not there is a new version of SecAI
    '''
    UPDATE_CHECK = True
    SPIDER_FLOAD = os.path.join(STATIC_FOLDER, "website")


class TestingConfig(Config):
    SECRET_KEY = 'AAAAAAAAAAAAAAAAAAAA'
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TESTING_DATABASE_URL') or 'sqlite://'
    SERVER_NAME = 'localhost'
    UPDATE_CHECK = False
    CACHE_REDIS_URL = None
    CACHE_TYPE = 'simple'
