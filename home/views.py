from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.views import View


class Test(View):
    def get(self, request):
        return HttpResponse('okay')
