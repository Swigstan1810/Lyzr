<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ClientDataController;



Route::get('/', function () {
    return view('welcome');
});

Route::post('/client-data', [ClientDataController::class, 'store']);