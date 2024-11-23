<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class ClientDataController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'risk_tolerance' => 'required|string',
            'financial_goals' => 'required|array|min:1',
            'timeline' => 'required|string',
            'financial_standing.income' => 'required|numeric|min:0',
            'financial_standing.expenses' => 'required|numeric|min:0',
            'financial_standing.savings' => 'required|numeric|min:0',
            'financial_standing.debt' => 'required|numeric|min:0',
        ]);

        return response()->json(['message' => 'Data saved successfully', 'data' => $validated]);
    }
}

