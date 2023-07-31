
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from detection import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(('detection.urls','detection'), namespace = 'detection'))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)