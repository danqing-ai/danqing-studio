# backend/core/__init__.py
from .interfaces import *
from .container import Container, get_container, register_services

__all__ = ['Container', 'get_container', 'register_services']
