# decorators for checking user roles before allowing access to certain views.
from rest_framework.response import Response
from functools import wraps

def role_required(required_roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(view, request, *args, **kwargs):
            if not request.user.is_authenticated or request.user.role not in required_roles:
                return Response({'error': 'Unauthorized access.'}, status=403)
            return view_func(view, request, *args, **kwargs)
        return _wrapped_view
    return decorator
