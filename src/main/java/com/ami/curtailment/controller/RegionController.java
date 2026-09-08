package com.ami.curtailment.controller;

import com.ami.curtailment.domain.Region;
import com.ami.curtailment.repository.RegionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/regions")
@RequiredArgsConstructor
public class RegionController {

    private final RegionRepository regionRepository;

    @GetMapping
    public List<Region> getAll() {
        return regionRepository.findAll();
    }

    @PostMapping
    public Region create(@RequestBody Region region) {
        return regionRepository.save(region);
    }
}
