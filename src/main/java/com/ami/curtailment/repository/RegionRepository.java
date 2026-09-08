package com.ami.curtailment.repository;

import com.ami.curtailment.domain.EnergySource;
import com.ami.curtailment.domain.Region;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RegionRepository extends JpaRepository<Region, Long> {
    List<Region> findByEnergySource(EnergySource energySource);
    Region findByName(String name);
}
