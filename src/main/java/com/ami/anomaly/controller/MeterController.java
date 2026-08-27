package com.ami.anomaly.controller;

import com.ami.anomaly.domain.Meter;
import com.ami.anomaly.repository.MeterRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/meters")
@RequiredArgsConstructor
public class MeterController {

    private final MeterRepository meterRepository;

    @GetMapping
    public List<Meter> getAll() {
        return meterRepository.findAll();
    }

    @PostMapping
    public Meter create(@RequestBody Meter meter) {
        return meterRepository.save(meter);
    }
}
